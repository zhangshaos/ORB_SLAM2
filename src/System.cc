/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

//主进程的实现文件

#include <thread>

#include <pangolin/pangolin.h>	// 可视化界面
#include <glog/logging.h>
#include <happly/happly.h>      // Save ply file
#include <fmt/format.h>         // fmt::format()

#include "KeyFrame.h"
#include "MapPoint.h"
#include "Map.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Converter.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Viewer.h"
#include "System.h"


namespace ORB_SLAM2
{

//系统的构造函数，将会启动其他的线程
System::System(const string &strVocFile,				//词典文件路径
			         const string &strSettingsFile,   //配置文件路径
               bool bUseViewer) 					      //是否使用可视化界面
               : mTrackingState{Tracking::State::SYSTEM_NOT_READY}
{
    // Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       LOG(ERROR) << "Failed to open settings file at: " << strSettingsFile << '\n';
       exit(-1);
    }

    // Load ORB Vocabulary
    LOG(INFO) << "Loading ORB Vocabulary. This could take a while...\n";

    // 建立一个新的ORB字典
    mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);
    if(!bVocLoad)
    {
        LOG(ERROR) << "Wrong path to vocabulary. "
                   << "Failed to open at: " << strVocFile << '\n';
        exit(-1);
    }
    LOG(INFO) << "Vocabulary loaded!\n";

    // Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    // Create the Map
    mpMap = new Map();

    // 这里的帧绘制器和地图绘制器将会被可视化的Viewer所使用
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    // 在本主进程中初始化追踪线程
    mpTracker = new Tracking(this,
                             mpVocabulary,				//字典
                             mpFrameDrawer, 			//帧绘制器
    						             mpMapDrawer,				  //地图绘制器
                             mpMap, 					    //地图
                             mpKeyFrameDatabase, 	//关键帧地图
                             strSettingsFile 		  //设置文件路径
                             );

    // 初始化局部建图线程并运行
    mpLocalMapper = new LocalMapping(mpMap, true);
    {
        std::thread localMapping(&ORB_SLAM2::LocalMapping::Run, mpLocalMapper);
        mtLocalMapping.swap(localMapping);
    }

    // 初始化回环检测线程并运行
    mpLoopCloser = new LoopClosing(mpMap, 						  //地图
                                   mpKeyFrameDatabase, 	//关键帧数据库
                                   mpVocabulary);       //ORB词典
    {
        std::thread loopClosing(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);
        mtLoopClosing.swap(loopClosing);
    }

    // 初始化可视化线程并执行
    if(bUseViewer)
    {
        mpViewer = new Viewer(this,
        					            mpFrameDrawer,	  //帧绘制器
        					            mpMapDrawer,		  //地图绘制器
        					            mpTracker,		    //追踪器
                              strSettingsFile);	//配置文件的访问路径
        {
            std::thread viewer(&Viewer::Run, mpViewer);
            mtViewer.swap(viewer);
        }
        mpTracker->SetViewer(mpViewer);
    }

    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}


System::~System()
{
    delete mpVocabulary;
    delete mpKeyFrameDatabase;
    delete mpMap;
    delete mpTracker;
    delete mpLocalMapper;
    delete mpLoopCloser;
    delete mpFrameDrawer;
    delete mpMapDrawer;
    delete mpViewer;
}


//判断是否地图有较大的改变
bool System::MapChanged()
{
    static int n=0;
    //其实整个函数功能实现的重点还是在这个GetLastBigChangeIdx函数上
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

//准备执行复位
void System::Reset()
{
    mbReset = true;
}

//退出
void System::Shutdown()
{
	  //对局部建图线程和回环检测线程发送终止请求
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            this_thread::sleep_for(5ms);
    }
    while(!mpLocalMapper->isFinished() ||
    	    !mpLoopCloser->isFinished()  ||
          mpLoopCloser->isRunningGBA())
    {
        this_thread::sleep_for(5ms);
    }
    if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

//获取追踪器状态
int System::GetTrackingState()
{
    return mTrackingState;
}

//获取追踪到的地图点（其实实际上得到的是一个指针）
vector<MapPoint*> System::GetTrackedMapPoints()
{
    return mTrackedMapPoints;
}

//获取追踪到的关键帧的点
vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    return mTrackedKeyPointsUn;
}

bool System::SaveMap(const string &filename)
{
  auto mapPoints = mpMap->GetAllMapPoints();
  if (mapPoints.empty())
    return false;
  happly::PLYData plyFile;
  std::vector<std::array<double, 3>> vertexesPosition;
  for (auto p : mapPoints)
    if (p && !p->isBad())
    {
      cv::Mat wPos = p->GetWorldPos();
      double x = wPos.at<float>(0), y = wPos.at<float>(1), z = wPos.at<float>(2);
      Eigen::Vector3d eigenWPos{x, y, z};
      eigenWPos = mRealTransform * eigenWPos;
      x = eigenWPos.x();
      y = eigenWPos.y();
      z = eigenWPos.z();
      vertexesPosition.emplace_back(std::array<double,3>{x, y, z});
    }
  plyFile.addVertexPositions(vertexesPosition);
  plyFile.write(filename, happly::DataFormat::Binary);
  return true;
}


int System::TrackMonocularWithPose(const cv::Mat &im, double timestamp,
                                   const Sophus::SE3d &poseTcw,
                                   const std::string& imName)
{
  mInPoseTcw = poseTcw;
  mInImage = im;
  mInImageName = imName;

  // Check reset
  if(mbReset)
  {
    mpTracker->Reset();
    mbReset = false;
  }

  // 获取相机位姿的估计结果
  auto state = mpTracker->trackImageWithPose(im, timestamp, Converter::toCvMat(poseTcw), imName.c_str());
  mTrackingState = state;
  mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
  mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
  return state;
}

bool System::SaveTrackedMap(const std::string &filePath)
{
  std::ofstream trackedFile(filePath);
  if (!trackedFile.is_open())
  {
    LOG(INFO) << fmt::format("Write to {} failed. It is not open.\n", filePath);
    return false;
  }

  auto keyPoints = GetTrackedKeyPointsUn();
  auto mapPoints = GetTrackedMapPoints();
  assert(keyPoints.size() == mapPoints.size());
  // Save the results as .ply file.
  happly::PLYData plyFile;
  std::vector<std::array<double, 3>> vertexesPos;
  std::vector<std::array<uchar, 3>> vertexesColor;
  std::vector<float> vertexesPtX;
  std::vector<float> vertexesPtY;
  std::vector<uchar> vertexesPtOctave;
  const size_t N = keyPoints.size();
  for (size_t i=0; i<N; ++i)
  {
    MapPoint* p = mapPoints[i];
    const cv::KeyPoint& kpt = keyPoints[i];
    if (p && !p->isBad())
    {
      Eigen::Vector3d wPos = Converter::toVector3d(p->GetWorldPos());
      Eigen::Vector3d camPos = mInPoseTcw * wPos;
      vertexesPos.emplace_back(std::array<double,3>{ camPos.x(), camPos.y(), camPos.z() });
      vertexesPtX.emplace_back(kpt.pt.x);
      vertexesPtY.emplace_back(kpt.pt.y);
      vertexesPtOctave.emplace_back(kpt.octave);
      const auto& ptColor = mInImage.at<cv::Vec3b>(kpt.pt.y, kpt.pt.x);
      vertexesColor.emplace_back(std::array<uchar,3>{ptColor[2], ptColor[1], ptColor[0]});  // r, g, b
    }
  }
  const size_t M = vertexesPos.size();
  if (M <= 0)
    return false;
  // 写入当前帧信息
  uint32_t fID = mpTracker->mCurrentFrame.mnId;
  cv::Mat Tcw = mpTracker->mCurrentFrame.mTcw * Converter::toCvMat(mRealTransform.inverse());
  std::vector<float> TcwParams;
  for (int i=0; i<Tcw.rows; ++i)
    for (int j=0; j<Tcw.cols; ++j)
      TcwParams.emplace_back(Tcw.at<float>(i, j));
  const auto& K = mpTracker->mCurrentFrame.mK;
  std::vector<float> KParams;
  for (int i=0; i<K.rows; ++i)
    for (int j=0; j<K.cols; ++j)
      KParams.emplace_back(K.at<float>(i, j));
  plyFile.addElement("frame", 1);
  plyFile.getElement("frame").addProperty<uint32_t>("ID", {fID});
  plyFile.getElement("frame").addListProperty<float>("Tcw", {TcwParams});
  plyFile.getElement("frame").addListProperty<float>("K", {KParams});
  plyFile.addVertexPositions(vertexesPos);
  plyFile.addVertexColors(vertexesColor);
  plyFile.getElement("vertex").addProperty<float>("ix", vertexesPtX);
  plyFile.getElement("vertex").addProperty<float>("iy", vertexesPtY);
  plyFile.getElement("vertex").addProperty<uchar>("octave", vertexesPtOctave);
  plyFile.write(trackedFile);
  return true;
}

} //namespace ORB_SLAM
