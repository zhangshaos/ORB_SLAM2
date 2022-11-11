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


#include <iostream>
#include <cmath>
#include <mutex>

#include <opencv2/core/core.hpp>
#include <glog/logging.h>
#include <fmt/format.h>

#include "Converter.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "Initializer.h"
#include "FrameDrawer.h"
#include "Viewer.h"
#include "MapPoint.h"
#include "Map.h"
#include "MapDrawer.h"
#include "KeyFrameDatabase.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "System.h"
#include "Tracking.h"
#include "Optimizer.h"
#include "PnPsolver.h"


using namespace std;

// 程序中变量名的第一个字母如果为"m"则表示为类中的成员变量，member
// 第一个、第二个字母:
// "p" 表示指针数据类型
// "n" 表示int类型
// "b" 表示bool类型
// "s" 表示set类型
// "v" 表示vector数据类型
// 'l' 表示list数据类型
// "KF" 表示KeyFrame数据类型

namespace ORB_SLAM2
{


// 构造函数
Tracking::Tracking(
  System *pSys,                       //系统实例
  ORBVocabulary* pVoc,                //BOW字典
  FrameDrawer *pFrameDrawer,          //帧绘制器
  MapDrawer *pMapDrawer,              //地图点绘制器
  Map *pMap,                          //地图句柄
  KeyFrameDatabase* pKFDB,            //关键帧产生的词袋数据库
  const string &strSettingPath        //配置文件路径
  ):
  mState(NO_IMAGES_YET),          //当前系统还没有准备好
  mpORBVocabulary(pVoc),
  mpKeyFrameDB(pKFDB),
  mpInitializer(nullptr),         //暂时给地图初始化器设置为空指针
  mpSystem(pSys),
  mpViewer(nullptr),              //注意可视化的查看器是可选的，因为ORB-SLAM2最后是被编译成为一个库，所以对方人拿过来用的时候也应该有权力说我不要可视化界面（何况可视化界面也要占用不少的CPU资源）
  mpFrameDrawer(pFrameDrawer),
  mpMapDrawer(pMapDrawer),
  mpMap(pMap),
  mnLastKeyFrameId(0),
  mnLastRelocFrameId(0),           //恢复为0,没有进行这个过程的时候的默认值
  mLastProcessedState(State::SYSTEM_NOT_READY),
  mpLocalMapper(nullptr),
  mpLoopClosing(nullptr),
  mpReferenceKF(nullptr),
  mnMatchesInliers(0)
{
  // Load camera parameters from settings file
  // Step 1 从配置文件中加载相机参数
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
  float fx = fSettings["Camera.fx"];
  float fy = fSettings["Camera.fy"];
  float cx = fSettings["Camera.cx"];
  float cy = fSettings["Camera.cy"];

  //     |fx  0   cx|
  // K = |0   fy  cy|
  //     |0   0   1 |
  //构造相机内参矩阵
  cv::Mat K = cv::Mat::eye(3,3,CV_32F);
  K.at<float>(0,0) = fx;
  K.at<float>(1,1) = fy;
  K.at<float>(0,2) = cx;
  K.at<float>(1,2) = cy;
  K.copyTo(mK);

  // 图像矫正系数
  // [k1 k2 p1 p2 k3]
  cv::Mat DistCoef(4,1,CV_32F);
  DistCoef.at<float>(0) = fSettings["Camera.k1"];
  DistCoef.at<float>(1) = fSettings["Camera.k2"];
  DistCoef.at<float>(2) = fSettings["Camera.p1"];
  DistCoef.at<float>(3) = fSettings["Camera.p2"];
  const float k3 = fSettings["Camera.k3"];
  //有些相机的畸变系数中会没有k3项
  if(k3!=0)
  {
    DistCoef.resize(5);
    DistCoef.at<float>(4) = k3;
  }
  DistCoef.copyTo(mDistCoef);

  float fps = fSettings["Camera.fps"];
  if(fps == 0)
    fps = 30;

  // Max/Min Frames to insert keyframes and to check relocalisation
  mMinFrames = 0;
  mMaxFrames = (int)fps;

  //输出
  LOG(INFO)
  << "Camera Parameters:\n"
  << "- fx: " << fx << '\n'
  << "- fy: " << fy << '\n'
  << "- cx: " << cx << '\n'
  << "- cy: " << cy << '\n'
  << "- k1: " << DistCoef.at<float>(0) << '\n'
  << "- k2: " << DistCoef.at<float>(1) << '\n'
  << "- p1: " << DistCoef.at<float>(2) << '\n'
  << "- p2: " << DistCoef.at<float>(3) << '\n'
  << "- fps: " << fps << '\n';

  // 1:RGB 0:BGR
  int nRGB = fSettings["Camera.RGB"];
  mbRGB = nRGB;

  if(mbRGB)
    LOG(INFO) << "- color order: RGB (ignored if grayscale)" << '\n';
  else
    LOG(INFO) << "- color order: BGR (ignored if grayscale)" << '\n';

  // Load ORB parameters

  // Step 2 加载ORB特征点有关的参数,并新建特征点提取器

  // 每一帧提取的特征点数 1000
  int nFeatures = fSettings["ORBextractor.nFeatures"];
  // 图像建立金字塔时的变化尺度 1.2
  float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
  // 尺度金字塔的层数 8
  int nLevels = fSettings["ORBextractor.nLevels"];
  // 提取fast特征点的默认阈值 20
  int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
  // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
  int fMinThFAST = fSettings["ORBextractor.minThFAST"];

  // Tracking过程都会用到 mpORBextractorLeft 作为特征点提取器
  mpORBextractorLeft = new ORBextractor(
    nFeatures,      //参数的含义还是看上面的注释吧
    fScaleFactor,
    nLevels,
    fIniThFAST,
    fMinThFAST);

  // 在单目初始化的时候，会用 mpIniORBextractor 来作为特征点提取器
  mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

  LOG(INFO)
  << "ORB Extractor Parameters:\n"
  << "- Number of Features: " << nFeatures << '\n'
  << "- Scale Levels: " << nLevels << '\n'
  << "- Scale Factor: " << fScaleFactor << '\n'
  << "- Initial Fast Threshold: " << fIniThFAST << '\n'
  << "- Minimum Fast Threshold: " << fMinThFAST << '\n';
}


Tracking::State
Tracking::trackImageWithPose(const cv::Mat &im,
                             double timestamp,
                             const cv::Mat &poseTcw,
                             const char* imName)
{
  mImGray = im;

  // Step 1 ：将彩色图像转为灰度图像
  if(mImGray.channels()==3)
  {
    if(mbRGB)
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
  }
  else if(mImGray.channels()==4)
  {
    if(mbRGB)
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
  }

  // Step 2 ：构造Frame
  if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET) // 没有成功初始化的前一个状态就是NO_IMAGES_YET
    mCurrentFrame = Frame(
      mImGray,
      timestamp,
      mpIniORBextractor,  // 初始化ORB特征点提取器会提取2倍的指定特征点数目
      mpORBVocabulary,
      mK,
      mDistCoef);
  else
    mCurrentFrame = Frame(
      mImGray,
      timestamp,
      mpORBextractorLeft, // 正常运行的时的ORB特征点提取器，提取指定数目特征点
      mpORBVocabulary,
      mK,
      mDistCoef);

  /*********************************************************************************
   * 设置（初始）位姿。
   * 我们认为无人机的位姿是准确的，在大部分情况下都不优化位姿，只有回环检测Global BA时才会优化它。
   *********************************************************************************/
  mCurrentFrame.SetPose(poseTcw);

  // Step 3 ：跟踪，包含两部分：估计运动、跟踪局部地图
  if(mState==NO_IMAGES_YET) // 如果图像复位过、或者第一次运行，则为NO_IMAGE_YET状态
    mState = NOT_INITIALIZED;

  // mLastProcessedState 存储了Tracking最新的状态，用于FrameDrawer中的绘制
  mLastProcessedState = mState;

  do {
    // 地图更新时加锁，保证地图不会发生变化。
    // 这样子会不会影响地图的实时更新?
    // 主要耗时在构造帧中特征点的提取和匹配部分,在那个时候地图是没有被上锁的,有足够的时间更新地图
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
      // 单目初始化
      Initialization();

      // 更新帧绘制器中存储的最新状态
      mpFrameDrawer->Update(this);

      if(mState!=OK) // 这个状态量在上面的初始化函数中被更新
        break;
    }
    else
    {
      bool bOK = false;
      // Step 2：跟踪进入正常SLAM模式，有地图更新
      if(mState==OK)
      {
        // Step 2.1 检查并更新上一帧被替换的 MapPoints，局部建图线程则可能会对原有的地图点进行替换。
        CheckReplacedMapPointsInLastFrame();
        // Step 2.2 追踪（特征点和地图点）
        bOK = TrackWithInitialPose();
        if (!bOK)
          // Step 2.3 追踪
          bOK = TrackWithReferenceKF();
        if (!bOK)
          // 丢失追踪了
          LOG(INFO) << "Lost tracking with Frame " << mCurrentFrame.mnId << "...\n"
                    << "Its image file name is " << (imName ? imName : "unknown");
      } else
      {
        // todo: 当追踪失败后，就地初始化一个新地图，保存旧地图数据
        bOK = Relocalization();
        if (bOK)
          LOG(INFO) << "Relocating is successful!";
        else
          LOG(INFO) << "Relocating is failed for Frame " << mCurrentFrame.mnId << '!';
      }

      mCurrentFrame.mpReferenceKF = mpReferenceKF;

      // Step 3：在跟踪得到当前帧初始姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
      // 这里搜索局部关键帧、局部地图点，和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
      if (bOK)
        bOK = TrackLocalMap();

      if(bOK)
        mState = OK;
      else
        mState = LOST;

      // Step 4：更新显示线程中的图像、特征点、地图点等信息
      mpFrameDrawer->Update(this);

      if(bOK) // 只有在成功追踪时才考虑生成关键帧的问题
      {
        // 更新显示中的位姿
        {
          //cv::Mat pose = mCurrentFrame.getPose().clone();
          //LOG(INFO) << "Old Pose:\n" << pose;
          //pose.at<float>(0, 3) /= 1000;
          //pose.at<float>(1, 3) /= 1000;
          //pose.at<float>(2, 3) /= 1000;
          //LOG(INFO) << "New Pose:\n" << pose;
          mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.getPose());
        }

        // Step 6：清除观测不到的地图点
        for(int i=0; i<mCurrentFrame.N; i++)
        {
          MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
          if(pMP && pMP->Observations() < 1)
            mCurrentFrame.mvpMapPoints[i] = nullptr;
        }

        // Step 8：检测并插入关键帧，对于双目或RGB-D会产生新的地图点
        if(NeedNewKeyFrame())
        {
          CreateNewKeyFrame();
          LOG(INFO) << fmt::format("Create a new Key Frame {} from Frame {}.",
                                   mpReferenceKF->mnId, mpReferenceKF->mnFrameId);
        }
      }

      // Step 10 如果初始化后不久就跟踪失败，并且relocation也没有搞定，只能重新Reset
      if(mState==LOST && mpMap->KeyFramesInMap() <= 5)
      {
        LOG(INFO) << "Track lost soon after initialisation, reseting..." << '\n';
        mpSystem->Reset();
        break;
      }

      // 确保已经设置了参考关键帧
      if(!mCurrentFrame.mpReferenceKF)
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

      // 保存上一帧的数据,当前帧变上一帧
      mLastFrame = Frame(mCurrentFrame);
    }
  } while(0);

  return mState;
}


// 设置局部建图器
void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
  mpLocalMapper = pLocalMapper;
}

// 设置回环检测器
void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
  mpLoopClosing = pLoopClosing;
}

// 设置可视化查看器
void Tracking::SetViewer(Viewer *pViewer)
{
  mpViewer = pViewer;
}


/*
 * @brief 单目的地图初始化
 *
 * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 * 得到初始两帧的匹配、相对运动、初始MapPoints
 *
 * Step 1：（未创建）得到用于初始化的第一帧，初始化需要两帧
 * Step 2：（已创建）如果当前帧特征点数大于100，则得到用于单目初始化的第二帧
 * Step 3：在mInitialFrame与mCurrentFrame中找匹配的特征点对
 * Step 4：如果初始化的两帧之间的匹配点太少，重新初始化
 * Step 5：通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
 * Step 6：删除那些无法进行三角化的匹配点
 * Step 7：将三角化得到的3D点包装成MapPoints
 */
void Tracking::Initialization()
{
  // Step 1 如果单目初始器还没有被创建，则创建。后面如果重新初始化时会清掉这个
  if(mpInitializer == nullptr)
  {
    // 单目初始帧的特征点数必须大于100
    if(mCurrentFrame.getKeyPoints().size() > 100)
    {
      // 初始化需要两帧，分别是mInitialFrame，mCurrentFrame
      mInitialFrame = Frame(mCurrentFrame);
      mLastFrame = Frame(mCurrentFrame);
      mpInitializer =  new Initializer(1.0);
      // 初始化为-1 表示没有任何匹配。这里面存储的是匹配的点的id
      fill(mvInitialMatches.begin(), mvInitialMatches.end(), -1);
      return;
    }
  }
  else // 如果单目初始化器已经被创建
  {
    // Step 2 如果当前帧特征点数太少（不超过100），则重新构造初始器
    // 只有连续两帧的特征点个数都大于100时，才能继续进行初始化过程
    if(mCurrentFrame.getKeyPoints().size() <= 100)
    {
      delete mpInitializer;
      mpInitializer = nullptr;
      fill(mvInitialMatches.begin(), mvInitialMatches.end(), -1);
      return;
    }

    // Step 3 在 mInitialFrame 与 mCurrentFrame 中找匹配的特征点对
    ORBmatcher matcher(0.9,   //最佳的和次佳特征点评分的比值阈值，这里是比较宽松的，跟踪时一般是0.7
                       true); //检查特征点的方向

    // 对 mInitialFrame 和 mCurrentFrame 进行特征点匹配
    vector<cv::Point2f> vMatchedKeyPoints;
    for(const auto& kp : mInitialFrame.getKeyPoints())
      vMatchedKeyPoints.emplace_back(kp.pt);
    int nMatch = matcher.SearchForInitialization(
      mInitialFrame, mCurrentFrame,   //初始化时的参考帧和当前帧
      vMatchedKeyPoints,              //参考帧的特征点坐标，初始化存储的是mInitialFrame中特征点坐标，匹配后存储的是匹配好的当前帧的特征点坐标
      mvInitialMatches,               //保存参考帧F1中特征点是否匹配上，index保存是F1对应特征点索引，值保存的是匹配好的F2特征点索引
      100);                           //搜索窗口大小

    // Step 4 验证匹配结果，如果初始化的两帧之间的匹配点太少，重新初始化
    if(nMatch < 100)
    {
      LOG(INFO)
      << "Try initialize with Frame " << mCurrentFrame.mnId
      << " with Frame " << mInitialFrame.mnId
      << "...\nBut keypoint matches: " << nMatch;
      delete mpInitializer;
      mpInitializer = nullptr;
      return;
    }

    // Step 5 初始化 MapPoints
    vector<cv::Point3f> vP3D;
    vector<bool> vbTriangulated; // Triangulated Correspondences (mvInitialMatches)
    if(mpInitializer->initializeMapPoints(mInitialFrame, mCurrentFrame, mvInitialMatches,
                                          vP3D, vbTriangulated))
    {
      // Step 6 初始化成功后，删除那些无法进行三角化的匹配点
      for(size_t i=0, iend=mvInitialMatches.size(); i < iend; i++)
        if(mvInitialMatches[i] >= 0 && !vbTriangulated[i])
        {
          mvInitialMatches[i] = -1;
          nMatch--;
        }
      // Step 8 创建初始化地图点 MapPoints
      CreateInitialMap(vP3D);
    } //当初始化成功的时候进行
  } //如果单目初始化器已经被创建
}

// 生成初始地图，由 Initialization() 调用
void Tracking::CreateInitialMap(const std::vector<cv::Point3f>& vIniP3D)
{
  // Create KeyFrames 认为单目初始化时候的参考帧和当前帧都是关键帧
  KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);  // 第一帧
  KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);  // 第二帧

  // Step 1 将初始关键帧,当前关键帧的描述子转为BoW
  pKFini->ComputeBoW();
  pKFcur->ComputeBoW();

  // Insert KFs in the map
  // Step 2 将关键帧插入到地图
  mpMap->AddKeyFrame(pKFini);
  mpMap->AddKeyFrame(pKFcur);

  // Create MapPoints and asscoiate to keyframes
  // Step 3 用初始化得到的3D点来生成地图点MapPoints
  //  vIniMatches[i] 表示初始化两帧特征点匹配关系。
  //  具体解释：i表示帧1中关键点的索引值，vMatches12[i]的值为帧2的关键点索引值,没有匹配关系的话，vMatches12[i]值为 -1
  for(size_t i=0; i < mvInitialMatches.size(); i++)
  {
    // 没有匹配，跳过
    if(mvInitialMatches[i] < 0)
      continue;

    //Create MapPoint.
    // 用三角化点初始化为空间点的世界坐标
    cv::Mat worldPos(vIniP3D[i]);

    // Step 3.1 用3D点构造MapPoint
    MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);

    // Step 3.2 为该MapPoint添加属性：
    // a.观测到该MapPoint的关键帧
    // b.该MapPoint的描述子
    // c.该MapPoint的平均观测方向和深度范围

    // 表示该KeyFrame的2D特征点和对应的3D地图点
    pKFini->AddMapPoint(pMP,i);
    pKFcur->AddMapPoint(pMP, mvInitialMatches[i]);

    // a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
    pMP->AddObservation(pKFini,i);
    pMP->AddObservation(pKFcur, mvInitialMatches[i]);

    // b.从众多观测到该MapPoint的特征点中挑选最有代表性的描述子
    pMP->ComputeDistinctiveDescriptors();
    // c.更新该MapPoint平均观测方向以及观测距离的范围
    pMP->UpdateNormalAndDepth();

    // Fill Current Frame structure
    // vIniMatches下标i表示在初始化参考帧中的特征点的序号
    // vIniMatches[i]是初始化当前帧中的特征点的序号
    mCurrentFrame.mvpMapPoints[mvInitialMatches[i]] = pMP;

    // Add to Map
    mpMap->AddMapPoint(pMP);
  }

  // Update Connections
  // Step 3.3 更新关键帧间的连接关系
  // 在3D点和关键帧之间建立边，每个边有一个权重，边的权重是该关键帧与当前帧公共3D点的个数
  pKFini->UpdateConnections();
  pKFcur->UpdateConnections();

  // Bundle Adjustment
  LOG(INFO) << "Create initial Map with " << mpMap->MapPointsInMap() << " points." << '\n';

  // Step 4 全局BA优化，同时优化所有三维点
  Optimizer::GlobalBundleAdjustemnt(mpMap, 20, nullptr, 0, true, {pKFini->mnId, pKFcur->mnId});

  // 两个条件,一个是平均深度要大于0, 另外一个是在当前帧中被观测到的地图点的数目应该大于100
  if(pKFcur->TrackedMapPoints(1)<100)
  {
    LOG(INFO) << "Wrong CreateInitialMap(), reseting...\n";
    Reset();
    return;
  }

  //  Step 8 将关键帧插入局部地图，更新归一化后的位姿、局部地图点
  mpLocalMapper->InsertKeyFrame(pKFini);
  mpLocalMapper->InsertKeyFrame(pKFcur);

  mCurrentFrame.SetPose(pKFcur->GetPose());
  mnLastKeyFrameId = mCurrentFrame.mnId;

  mvpLocalKeyFrames.push_back(pKFcur);
  mvpLocalKeyFrames.push_back(pKFini);
  // 单目初始化之后，得到的初始地图中的所有点都是局部地图点
  mvpLocalMapPoints = mpMap->GetAllMapPoints();
  mpReferenceKF = pKFcur;
  mCurrentFrame.mpReferenceKF = pKFcur; // 也只能这样子设置了,毕竟是最近的关键帧

  mpMap->SetLocalMapPoints(mvpLocalMapPoints);
  {
    //cv::Mat pose = pKFcur->GetPose().clone();
    //LOG(INFO) << "Old Pose:\n" << pose;
    //pose.at<float>(0, 3) /= 1000;
    //pose.at<float>(1, 3) /= 1000;
    //pose.at<float>(2, 3) /= 1000;
    //LOG(INFO) << "New Pose:\n" << pose;
    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());
  }
  mpMap->mvpKeyFrameOrigins.push_back(pKFini);
  mLastFrame = Frame(mCurrentFrame);
  mState = OK; // 初始化成功，至此，初始化过程完成
}

/*
 * @brief 检查上一帧中的地图点是否需要被替换
 *
 * Local Mapping线程可能会将关键帧中某些地图点进行替换，由于tracking中需要用到上一帧地图点，所以这里检查并更新上一帧中被替换的地图点
 * @see LocalMapping::FusePointsInNeighbors()
 */
void Tracking::CheckReplacedMapPointsInLastFrame()
{
  for(int i =0; i<mLastFrame.N; i++)
  {
    MapPoint* pMP = mLastFrame.mvpMapPoints[i];
    // 如果这个地图点存在
    if(pMP)
    {
      // 获取其是否被替换,以及替换后的点
      // 这也是程序不直接删除这个地图点删除的原因
      MapPoint* pRep = pMP->GetReplaced();
      if(pRep)
      {
        // 然后替换一下
        mLastFrame.mvpMapPoints[i] = pRep;
      }
    }
  }
}


/**
 * @brief 用局部地图进行跟踪，进一步优化位姿
 *
 * 1. 更新局部地图，包括局部关键帧和关键点，
 *    局部地图包括：K1个关键帧、K2个临近关键帧和参考关键帧 + 由这些关键帧观测到的MapPoints
 * 2. 对局部MapPoints进行投影匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 *
 * Step 1：更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
 * Step 2：在局部地图中查找与当前帧匹配的MapPoints, 其实也就是对局部地图点进行跟踪
 * Step 3：更新局部所有MapPoints后对位姿再次优化
 * Step 4：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
 * Step 5：根据跟踪匹配数目及回环情况决定是否跟踪成功
 * @return true         跟踪成功
 * @return false        跟踪失败
 */
bool Tracking::TrackLocalMap()
{
  // We have an estimation of the camera pose and some map points tracked in the frame.
  // We retrieve the local map and try to find matches to points in the local map.

  // Update Local KeyFrames and Local Points
  // Step 1：用共视图更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
  UpdateLocalKeyFrames();
  UpdateLocalMapPoints();
  // 设置参考地图点用于绘图显示局部地图点（红色）
  mpMap->SetLocalMapPoints(mvpLocalMapPoints);

  // Step 2：筛选局部地图中新增的在视野范围内的地图点，投影到当前帧搜索匹配，得到更多的匹配关系
  int nNewMatches = ProjectLocalPointsToCurrentFrame();
  //LOG(INFO) << "New matched map points in TrackLocalMap(): " << nNewMatches;

  // Optimize Pose
  // Step 3：前面新增了更多的匹配关系，BA优化得到更准确的位姿
  // Optimizer::PoseOptimization(&mCurrentFrame);

  // Update MapPoints Statistics
  // Step 4：更新当前帧的地图点被观测程度，并统计跟踪局部地图后匹配数目
  mnMatchesInliers = 0;
  for(int i=0; i<mCurrentFrame.N; i++)
  {
    if(mCurrentFrame.mvpMapPoints[i])
    {
      // 由于当前帧的地图点可以被当前帧观测到，其被观测统计量加1
      mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
      // 如果该地图点被相机观测数目nObs大于0，匹配内点计数+1
      // nObs： 被观测到的相机数目，单目+1，双目或RGB-D则+2
      if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
        mnMatchesInliers++;
    }
  }
  LOG(INFO)
  << "Frame " << mCurrentFrame.mnId
  << " total Map point matches: " << mnMatchesInliers;

  // Decide if the tracking was succesful
  // More restrictive if there was a relocalization recently
  // Step 5：根据跟踪匹配数目及重定位情况决定是否跟踪成功
  // 如果最近刚刚发生了重定位,那么至少成功匹配50个点才认为是成功跟踪
  if(mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
    return false;

  // 如果是正常的状态话只要跟踪的地图点大于30个就认为成功了
  return mnMatchesInliers >= 30;
}

/**
 * @brief 判断当前帧是否需要插入关键帧
 *
 * Step 1：纯VO模式下不插入关键帧，如果局部地图被闭环检测使用，则不插入关键帧
 * Step 2：如果距离上一次重定位比较近，或者关键帧数目超出最大限制，不插入关键帧
 * Step 3：得到参考关键帧跟踪到的地图点数量
 * Step 4：查询局部地图管理器是否繁忙,也就是当前能否接受新的关键帧
 * Step 5：对于双目或RGBD摄像头，统计可以添加的有效地图点总数 和 跟踪到的地图点数量
 * Step 6：决策是否需要插入关键帧
 * @return true         需要
 * @return false        不需要
 */
bool Tracking::NeedNewKeyFrame()
{
  // Step 1：纯VO模式下不插入关键帧

  // If Local Mapping is freezed by a Loop Closure do not insert keyframes
  // Step 2：如果局部地图线程被闭环检测使用，则不插入关键帧
  if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
    return false;

  // 获取当前地图中的关键帧数目
  const int nKFs = mpMap->KeyFramesInMap();

  // Do not insert keyframes if not enough frames have passed from last relocalisation
  // mCurrentFrame.mnId 是当前帧的ID
  // mnLastRelocFrameId 是最近一次重定位帧的ID
  // mMaxFrames 等于图像输入的帧率
  //  Step 3：如果距离上一次重定位比较近，并且关键帧数目超出最大限制，不插入关键帧
  if( mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
    return false;

  // Tracked MapPoints in the reference keyframe
  // Step 4：得到参考关键帧跟踪到的地图点数量
  // UpdateLocalKeyFrames 函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧

  int nMinObs = nKFs <= 2 ? 2 : 3; // 地图点的最小观测次数
  // 参考关键帧地图点中观测的数目 >= nMinObs的地图点数目
  int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

  // Local Mapping accept keyframes?
  // Step 5：查询局部地图线程是否繁忙，当前能否接受新的关键帧
  bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

  // Step 7：决策是否需要插入关键帧
  // Step 7.1：设定比例阈值，当前帧和参考关键帧跟踪到点的比例，比例越大，越倾向于增加关键帧
  float thRefRatio = 0.9f; // 单目情况下插入关键帧的频率很高

  // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
  // Step 7.2：很长时间没有插入关键帧，可以插入
  const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;

  // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
  // Step 7.3：满足插入关键帧的最小间隔并且localMapper处于空闲状态，可以插入
  const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);

  // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
  // Step 7.5：和参考帧相比当前跟踪到的点太少 或者满足bNeedToInsertClose；同时跟踪到的内点还不能太少
  const bool c2 = (mnMatchesInliers < int(nRefMatches*thRefRatio) && mnMatchesInliers > 15);

  if((c1a || c1b) && c2)
  {
    // If the mapping accepts keyframes, insert keyframe.
    // Otherwise send a signal to interrupt BA
    // Step 7.6：local mapping 空闲时可以直接插入，不空闲的时候要根据情况插入
    if(bLocalMappingIdle)
    {
      // 可以插入关键帧
      return true;
    }
    else
    {
      mpLocalMapper->InterruptBA();
      // 对于单目情况,就直接无法插入关键帧了
      // 为什么这里对单目情况的处理不一样? 可能是单目关键帧相对比较密集
      return false;
    }
  }
  else
    // 不满足上面的条件,自然不能插入关键帧
    return false;
}

/**
 * @brief 创建新的关键帧
 * 对于非单目的情况，同时创建新的MapPoints
 *
 * Step 1：将当前帧构造成关键帧
 * Step 2：将当前关键帧设置为当前帧的参考关键帧
 * Step 3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
 */
void Tracking::CreateNewKeyFrame()
{
  // 如果局部建图线程关闭了,就无法插入关键帧
  if(!mpLocalMapper->SetNotStop(true))
    return;

  // Step 1：将当前帧构造成关键帧
  KeyFrame* pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

  // Step 2：将当前关键帧设置为当前帧的参考关键帧
  // 在 UpdateLocalKeyFrames 函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
  mpReferenceKF = pKF;
  mCurrentFrame.mpReferenceKF = pKF;

  // Step 4：插入关键帧
  // 关键帧插入到列表 mlNewKeyFrames中，等待local mapping线程临幸
  mpLocalMapper->InsertKeyFrame(pKF);

  // 插入好了，允许局部建图停止
  mpLocalMapper->SetNotStop(false);

  // 当前帧成为新的关键帧，更新
  mnLastKeyFrameId = mCurrentFrame.mnId;
}


/**
 * @brief 用局部地图点进行投影匹配，得到更多的匹配关系
 * 注意：局部地图点中已经是当前帧地图点的不需要再投影，只需要将此外的并且在视野范围内的点和当前帧进行投影匹配
 *
 * @return 返回新增的地图点匹配数量。
 */
int Tracking::ProjectLocalPointsToCurrentFrame()
{
  // Do not search map points already matched
  // Step 1：遍历当前帧的地图点，标记这些地图点不参与之后的投影搜索匹配
  for(MapPoint* &pMP : mCurrentFrame.mvpMapPoints)
    if(pMP)
    {
      if(pMP->isBad())
      {
        pMP = nullptr;
      }
      else
      {
        // 更新能观测到该点的帧数加1(被当前帧观测了)
        pMP->IncreaseVisible();
        // 标记该点被当前帧观测到
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        // 标记该点在后面搜索匹配时(ORBmatcher::SearchByProjection)不被投影，因为已经有匹配了
        pMP->mbNeedTrackInView = false;
      }
    }

  // Project points in frame and check its visibility
  // Step 2：判断所有局部地图点中除当前帧地图点外的点，是否在当前帧视野范围内
  int nToMatch = 0; // 准备进行投影匹配的点的数目
  for(auto pMP : mvpLocalMapPoints)
  {
    // 已经被当前帧观测到的地图点肯定在视野范围内，跳过
    if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
      continue;
    // 跳过坏点
    if(pMP->isBad())
      continue;
    // Project (this fills MapPoint variables for matching)
    // 判断地图点是否在在当前帧视野内
    if(mCurrentFrame.isInFrustum(pMP, 0.5))
    {
      // 观测到该点的帧数加1
      pMP->IncreaseVisible();
      // 只有在视野范围内的地图点才参与之后的投影匹配
      nToMatch++;
    }
  }

  // Step 3：如果需要进行投影匹配的点的数目大于0，就进行投影匹配，增加更多的匹配关系
  int nMatched = 0;
  if(nToMatch>0)
  {
    ORBmatcher matcher(0.8);
    int th = 1;

    // If the camera has been relocalised recently, perform a coarser search
    // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
    if(mCurrentFrame.mnId < mnLastRelocFrameId+2)
      th = 5;

    // 投影匹配得到更多的匹配关系
    nMatched = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
  }
  return nMatched;
}


/*
 * @brief 更新局部关键点。先把局部地图清空，然后将局部关键帧的有效地图点添加到局部地图中
 */
void Tracking::UpdateLocalMapPoints()
{
  // Step 1：清空局部地图点
  mvpLocalMapPoints.clear();

  // Step 2：遍历局部关键帧 mvpLocalKeyFrames
  for(auto pKF : mvpLocalKeyFrames)
  {
    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();
    // step 2：将局部关键帧的地图点添加到 mvpLocalMapPoints
    for(auto pMP : vpMPs)
    {
      if(!pMP || pMP->isBad())
        continue;
      // 用该地图点的成员变量 mnTrackReferenceForFrame 记录当前帧的id
      // 表示它已经是当前帧的局部地图点了，可以防止重复添加局部地图点
      if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
        continue;
      mvpLocalMapPoints.push_back(pMP);
      pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
    }
  }
}

/**
 * @brief 跟踪局部地图函数里，更新局部关键帧
 * 方法是遍历当前帧的地图点，将观测到这些地图点的关键帧和相邻的关键帧及其父子关键帧，作为mvpLocalKeyFrames
 *
 * Step 1：遍历当前帧的地图点，记录所有能观测到当前帧地图点的关键帧
 * Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧包括以下3种类型
 *      类型1：能观测到当前帧地图点的关键帧，也称一级共视关键帧
 *      类型2：一级共视关键帧的共视关键帧，称为二级共视关键帧
 *      类型3：一级共视关键帧的子关键帧、父关键帧
 * Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
 */
void Tracking::UpdateLocalKeyFrames()
{
  // Each map point vote for the keyframes in which it has been observed
  // Step 1：遍历当前帧的地图点，记录所有能观测到当前帧地图点的关键帧
  map<KeyFrame*,int> keyframeCounter;
  for(int i=0; i<mCurrentFrame.N; i++)
  {
    if(mCurrentFrame.mvpMapPoints[i])
    {
      MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
      if(!pMP->isBad())
      {
        // 得到观测到该地图点的关键帧和该地图点在关键帧中的索引
        const map<KeyFrame*,size_t> observations = pMP->GetObservations();
        // 由于一个地图点可以被多个关键帧观测到,因此对于每一次观测,都对观测到这个地图点的关键帧进行累计投票
        for(const auto & observation : observations)
          // 这里的操作非常精彩！
          // map[key] = value，当要插入的键存在时，会覆盖键对应的原来的值。如果键不存在，则添加一组键值对
          // it->first 是地图点看到的关键帧，同一个关键帧看到的地图点会累加到该关键帧计数
          // 所以最后keyframeCounter 第一个参数表示某个关键帧，第2个参数表示该关键帧看到了多少当前帧(mCurrentFrame)的地图点，也就是共视程度
          keyframeCounter[observation.first]++;
      }
      else
      {
        mCurrentFrame.mvpMapPoints[i] = nullptr;
      }
    }
  }

  // 没有当前帧没有共视关键帧，返回
  if(keyframeCounter.empty())
    return;

  // Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有3种类型
  // 先清空局部关键帧
  mvpLocalKeyFrames.clear();
  // 先申请3倍内存，不够后面再加
  mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

  // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
  // Step 2.1 类型1：能观测到当前帧地图点的关键帧作为局部关键帧 （将邻居拉拢入伙）（一级共视关键帧）
  int max = 0;
  KeyFrame* pKFmax = nullptr; // 存储具有最多观测次数（max）的关键帧
  for(const auto & it : keyframeCounter)
  {
    KeyFrame* pKF = it.first;
    // 如果设定为要删除的，跳过
    if(pKF->isBad())
      continue;

    // 寻找具有最大观测数目的关键帧
    if(it.second > max)
    {
      max = it.second;
      pKFmax = pKF;
    }

    // 添加到局部关键帧的列表里
    mvpLocalKeyFrames.push_back(it.first);

    // 用该关键帧的成员变量 mnTrackReferenceForFrame 记录当前帧的id
    // 表示它已经是当前帧的局部关键帧了，可以防止重复添加局部关键帧
    pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
  }


  // Include also some not-already-included keyframes that are neighbors to already-included keyframes
  // Step 2.2 遍历一级共视关键帧，寻找更多的局部关键帧
  for(auto pKF : mvpLocalKeyFrames)
  {
    // Limit the number of keyframes
    // 处理的局部关键帧不超过80帧
    if(mvpLocalKeyFrames.size() > 80)
      break;

    // 类型2:一级共视关键帧的共视（前10个）关键帧，称为二级共视关键帧（将邻居的邻居拉拢入伙）
    // 如果共视帧不足10帧,那么就返回所有具有共视关系的关键帧
    const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
    // vNeighs 是按照共视程度从大到小排列
    for(auto pNeighKF : vNeighs)
      // mnTrackReferenceForFrame防止重复添加局部关键帧
      if(!pNeighKF->isBad() && pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
      {
        mvpLocalKeyFrames.push_back(pNeighKF);
        pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
        // 找到一个就直接跳出for循环？
        break;
      }

    // 类型3:将一级共视关键帧的子关键帧作为局部关键帧（将邻居的孩子们拉拢入伙）
    const set<KeyFrame*> spChilds = pKF->GetChilds();
    for(auto pChildKF : spChilds)
      if(!pChildKF->isBad() && pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
      {
        mvpLocalKeyFrames.push_back(pChildKF);
        pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
        // 找到一个就直接跳出for循环？
        break;
      }

    // 类型3:将一级共视关键帧的父关键帧（将邻居的父母们拉拢入伙）
    KeyFrame* pParent = pKF->GetParent();
    if(pParent && pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId) // mnTrackReferenceForFrame防止重复添加局部关键帧
    {
      mvpLocalKeyFrames.push_back(pParent);
      pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
    }
  }

  // Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
  if(pKFmax)
  {
    mpReferenceKF = pKFmax;
    mCurrentFrame.mpReferenceKF = mpReferenceKF;
  }
}


// 整个追踪线程执行复位操作
void Tracking::Reset()
{
  // 基本上是挨个请求各个线程终止

  if(mpViewer)
  {
    mpViewer->RequestStop();
    while(!mpViewer->isStopped())
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }
  LOG(INFO) << "System Reseting" << '\n';

  // Reset Local Mapping
  LOG(INFO) << "Reseting Local Mapper...";
  mpLocalMapper->RequestReset();
  LOG(INFO) << " done" << '\n';

  // Reset Loop Closing
  LOG(INFO) << "Reseting Loop Closing...";
  mpLoopClosing->RequestReset();
  LOG(INFO) << " done" << '\n';

  // Clear BoW Database
  LOG(INFO) << "Reseting Database...";
  mpKeyFrameDB->clear();
  LOG(INFO) << " done" << '\n';

  // Clear Map (this erase MapPoints and KeyFrames)
  mpMap->clear();

  //然后复位各种变量
  KeyFrame::nNextId = 0;
  Frame::nNextId = 0;
  mState = NO_IMAGES_YET;

  if(mpInitializer)
  {
    delete mpInitializer;
    mpInitializer = nullptr;
  }

  if(mpViewer)
    mpViewer->Release();
}


/**
 * 尝试使用当前帧初始化的位姿来进行当前帧特征-过去帧地图点匹配
 *
 * @return is a successful tracking.
 */
bool Tracking::TrackWithInitialPose()
{
  if (mCurrentFrame.getPose().empty())
    return false;
  fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
  ORBmatcher matcher(0.9, true);
  int th = 7;
  int nMatch = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, true);
  if (nMatch < 20)
    return false;
  int nBadMatch = CheckMatchesByProjection(5.991);
  return (nMatch - nBadMatch) >= 10;
}


/**
 * 尝试计算 Tracking 的参考KF地图点-当前帧特征匹配
 *
 * @return is a successful tracking.
 */
bool Tracking::TrackWithReferenceKF()
{
  mCurrentFrame.ComputeBoW();

  ORBmatcher matcher(0.7, true);
  vector<MapPoint*> vpMatchedMapPoints;
  // mpReferenceKF 什么时候设置的？
  // 1. Initialization()->CreateInitialMap()
  // 2. CreateNewKeyFrame()
  // 3. TrackLocalMap()->UpdateLocalKeyFrames()
  int nMatch = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMatchedMapPoints);
  if (nMatch < 15)
    return false;
  mCurrentFrame.mvpMapPoints = vpMatchedMapPoints;
  int nBadMatch = CheckMatchesByProjection(5.991);
  return (nMatch - nBadMatch) >= 10;
}


/**
 * 将当前帧已匹配的地图点重投影到图片上，计算误差error，
 * 并将error^2 > th2的地图点取消与特征点的匹配。
 *
 * 由 TrackWithInitialPose()、TrackWithReferenceKF()、Relocalization() 调用
 *
 * @return 错误匹配(error^2 > th2)的数量 if returnGood==false
 * @return 正确匹配的数量 if returnGood==true
 */
int Tracking::CheckMatchesByProjection(float th2, bool returnGood)
{
  // 构建投影矩阵
  cv::Mat Tcw = mCurrentFrame.getPose();
  cv::Mat K = mCurrentFrame.getCameraIntrincs();
  cv::Mat P(3, 4, CV_32F);
  Tcw.rowRange(0, 3).colRange(0, 4).copyTo(P);
  P = K * P;
  //LOG(INFO) << "Proj Matrix of current frame is:\n" << P;
  int nBad = 0, nGood = 0;
  for (int i=0; i<mCurrentFrame.N; ++i)
  {
    const cv::KeyPoint& kp = mCurrentFrame.getKeyPoints()[i];
    MapPoint *mp;
    if ((mp = mCurrentFrame.mvpMapPoints[i]))
    {
      cv::Mat wPosHomo(4, 1, CV_32F, cv::Scalar(1.0));
      mp->GetWorldPos().copyTo(wPosHomo.rowRange(0, 3));
      cv::Mat zuv = P * wPosHomo; // (u,v,1)*z
      float z = zuv.at<float>(2);
      float x = zuv.at<float>(0) / z, y = zuv.at<float>(1) / z;
      float err2 = (x-kp.pt.x)*(x-kp.pt.x)+(y-kp.pt.y)*(y-kp.pt.y);
      if (z <= 0 || err2 > th2)
      {
        mCurrentFrame.mvpMapPoints[i] = nullptr;
        mp->mbNeedTrackInView = false;
        if (z > 0)
          mp->mnLastFrameSeen = mCurrentFrame.mnId;
        ++nBad;
      } else
        ++nGood;
    }
  }
  return returnGood ? nGood : nBad;
}


/**
 * 在关键帧数据库中搜索关键帧进行KF地图点-当前帧特征匹配
 *
 * @return is a successful re-localization.
 */
bool Tracking::Relocalization()
{
  // Compute Bag of Words Vector
  mCurrentFrame.ComputeBoW();

  // Relocalization is performed when tracking is lost
  // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
  std::vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
  // 经过测试，上面这个经常只找到一个关键帧，因此添加一些最近的关键帧
  std::vector<KeyFrame*> candiKFs2 = mpMap->GetLastKeyFrames(mCurrentFrame.mnId, mMaxFrames);
  for (KeyFrame* KF : candiKFs2)
    vpCandidateKFs.emplace_back(KF);
  const int nKFs = vpCandidateKFs.size();
  if(nKFs <= 0)
  {
    LOG(INFO) << "Relocalization: Can not found similar KF for Frame " << mCurrentFrame.mnId;
    return false;
  }

  // 对某个候选项（关键帧）做ORB的地图点-当前帧特征点匹配
  ORBmatcher matcher(0.75, true);
  std::vector<std::vector<MapPoint*>> vvpMapPointMatches(nKFs); // 匹配上的地图点
  std::vector<bool> vbDiscarded(nKFs,false); // 标记那些当前帧没追踪上的关键帧
  int nCandi = 0;
  for(int i=0; i<nKFs; i++)
  {
    KeyFrame* pKF = vpCandidateKFs[i];
    if(pKF->isBad())
      vbDiscarded[i] = true;
    else
    {
      int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
      if(nmatches<15)
        vbDiscarded[i] = true;
      else
        ++nCandi;
    }
  }
  LOG(INFO)
  << "Relocalization: Found " << nCandi
  << " candidates in " << nKFs << " similar KFs"
  << " for Frame " << mCurrentFrame.mnId
  << "\nGlobal Map contains " << mpMap->KeyFramesInMap() << " KFs.";

  // 通过重投影误差过滤掉不好的匹配
  bool bFoundOne = false; // 找到一个最好的匹配候选项
  ORBmatcher matcher2(0.9, true);
  for(int i=0; i<nKFs; i++)
  {
    if(vbDiscarded[i])
      continue;

    // 计算当前帧地图点反投影误差，过滤不好的匹配
    std::set<MapPoint*> sFoundMapPoints;
    for(int j=0, jend=mCurrentFrame.N; j<jend; j++)
    {
      MapPoint *matchedMP = vvpMapPointMatches[i][j];
      if(matchedMP)
      {
        mCurrentFrame.mvpMapPoints[j]=matchedMP;
        sFoundMapPoints.insert(matchedMP);
      }
      else
        mCurrentFrame.mvpMapPoints[j]=nullptr;
    }
    int nGood = CheckMatchesByProjection(5.991, true);
    // 如果好匹配太少了，则试试：
    // 增大ORB匹配阈值，然后再反投影误差筛选
    if(nGood<50)
    {
      int nAdditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFoundMapPoints, 10, 100);
      if(nAdditional + nGood >= 50)
      {
        nGood = CheckMatchesByProjection(5.991, true);
        // 如果好匹配还是不够，则缩小匹配阈值，再试试
        if(nGood>30 && nGood<50)
        {
          sFoundMapPoints.clear();
          for(int j =0, jend=mCurrentFrame.N; j < jend; j++)
          {
            MapPoint *p;
            if ((p = mCurrentFrame.mvpMapPoints[j]))
              sFoundMapPoints.insert(p);
          }
          nAdditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFoundMapPoints, 3, 64);
          // 最后的核验重投影误差
          if(nGood + nAdditional >= 50)
            nGood = CheckMatchesByProjection(5.991, true);
        }
      }
    }

    // 如果找到了足够的匹配
    if(nGood>=50)
    {
      bFoundOne = true;
      break;
    } else
      LOG(INFO) << "Relocalization: Found " << nGood
      << " good matches w.r.t candidate KF " << vpCandidateKFs[i]->mnId;
  }

  if(bFoundOne)
  {
    mnLastRelocFrameId = mCurrentFrame.mnId;
    return true;
  }
  else
    return false;
}


} //namespace ORB_SLAM