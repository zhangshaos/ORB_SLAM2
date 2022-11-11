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

#include <mutex>

#include <pangolin/pangolin.h>
#include <glog/logging.h>

#include "MapPoint.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapDrawer.h"


namespace ORB_SLAM2
{

//构造函数
MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath):mpMap(pMap)
{
  //从配置文件中读取设置的
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

  mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
  mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
  mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
  mPointSize = fSettings["Viewer.PointSize"];
  mCameraSize = fSettings["Viewer.CameraSize"];
  mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];
  mCoordinateScale = fSettings["Viewer.CoordinateScale"];
}

void MapDrawer::DrawMapPoints()
{
  // 取出所有的地图点
  const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();
  // 取出mvpReferenceMapPoints，也即局部地图点
  const vector<MapPoint*> vpRefMPs = mpMap->GetLocalMapPoints();
  set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

  if(vpMPs.empty())
    return;

  // for AllMapPoints
  // 显示其他的地图点（不包括局部地图点），大小为2个像素，红色
  glPointSize(mPointSize);
  glBegin(GL_POINTS);
  glColor3f(1.0,0.0,0.0);
  for(auto mp : vpMPs)
  {
    // 不包括ReferenceMapPoints（局部地图点）
    if(mp->isBad() || spRefMPs.count(mp))
      continue;
    cv::Mat pos = mp->GetWorldPos();
    pos *= mCoordinateScale;
    glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
  }
  glEnd();

  // for ReferenceMapPoints
  // 显示局部地图点，大小为2个像素，黑色
  glPointSize(mPointSize);
  glBegin(GL_POINTS);
  glColor3f(0.0,0.0,0.0);
  for(auto spRefMP : spRefMPs)
  {
    if(spRefMP->isBad())
      continue;
    cv::Mat pos = spRefMP->GetWorldPos();
    pos *= mCoordinateScale;
    glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
  }
  glEnd();
}


void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
  //历史关键帧图标：宽度占总宽度比例为0.05
  const float &w = mKeyFrameSize;
  const float h = w*0.75;
  const float z = w*0.6;

  // step 1：取出所有的关键帧
  const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

  // step 2：显示所有关键帧图标
  //通过显示界面选择是否显示历史关键帧图标
  if(bDrawKF)
  {
    for(auto pKF : vpKFs)
    {
      //NOTICE 转置, OpenGL中的矩阵为列优先存储
      cv::Mat Twc = pKF->GetPoseInverse().t();
      Twc.at<float>(3, 0) *= mCoordinateScale;
      Twc.at<float>(3, 1) *= mCoordinateScale;
      Twc.at<float>(3, 2) *= mCoordinateScale;

      glPushMatrix();
      //（由于使用了glPushMatrix函数，因此当前帧矩阵为世界坐标系下的单位矩阵）
      //因为OpenGL中的矩阵为列优先存储，因此实际为Tcw，即相机在世界坐标下的位姿
      //NOTICE 竟然还可以这样写,牛逼牛逼
      glMultMatrixf(Twc.ptr<GLfloat>(0));

      //设置绘制图形时线的宽度
      glLineWidth(mKeyFrameLineWidth);
      //设置当前颜色为蓝色(关键帧图标显示为蓝色)
      glColor3f(0.0f,0.0f,1.0f);
      //用线将下面的顶点两两相连
      glBegin(GL_LINES);
      glVertex3f(0,0,0);
      glVertex3f(w,h,z);
      glVertex3f(0,0,0);
      glVertex3f(w,-h,z);
      glVertex3f(0,0,0);
      glVertex3f(-w,-h,z);
      glVertex3f(0,0,0);
      glVertex3f(-w,h,z);

      glVertex3f(w,h,z);
      glVertex3f(w,-h,z);

      glVertex3f(-w,h,z);
      glVertex3f(-w,-h,z);

      glVertex3f(-w,h,z);
      glVertex3f(w,h,z);

      glVertex3f(-w,-h,z);
      glVertex3f(w,-h,z);
      glEnd();

      glPopMatrix();
    }
  }

  // step 3：显示所有关键帧的Essential Graph (本征图)
  /**
   * 共视图中存储了所有关键帧的共视关系
   * 本征图中对边进行了优化,保存了所有节点,只存储了具有较多共视点的边,用于进行优化
   * 生成树则进一步进行了优化,保存了所有节点,但是值保存具有最多共视地图点的关键帧的边
   *
   */
  //通过显示界面选择是否显示关键帧连接关系
  if(bDrawGraph)
  {
    for(auto pKF : vpKFs)
    {
      cv::Mat Ow = pKF->GetCameraCenter();
      // Covisibility Graph (共视图)
      // step 3.1 共视程度比较高的共视关键帧用线连接
      // 遍历每一个关键帧，得到它们共视程度比较高的关键帧
      const vector<KeyFrame*> vCovKFs = pKF->GetCovisiblesByWeight(100);
      if(!vCovKFs.empty())
      {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,0.0f,0.0f,0.3f); // 设置共视图连接线为黑色，透明度为0.3f
        glBegin(GL_LINES);
        for (auto covKF: vCovKFs)
        {
          if (covKF->mnId < pKF->mnId)
            // 单向绘制
            continue;
          cv::Mat Ow2 = covKF->GetCameraCenter();
          glVertex3f(Ow.at<float>(0) * mCoordinateScale,
                     Ow.at<float>(1) * mCoordinateScale,
                     Ow.at<float>(2) * mCoordinateScale);
          glVertex3f(Ow2.at<float>(0) * mCoordinateScale,
                     Ow2.at<float>(1) * mCoordinateScale,
                     Ow2.at<float>(2) * mCoordinateScale);
        }
        glEnd();
      }

      // Spanning tree
      // step 3.2 连接最小生成树 (PS: 我觉得这里并不是权值最小,而是其中的边对于其他的图来讲是最少的)
      KeyFrame* pParent = pKF->GetParent();
      if(pParent)
      {
        cv::Mat Owp = pParent->GetCameraCenter();
        glLineWidth(2*mGraphLineWidth);
        glColor4f(0.f, 1.f, 0.f, 1.f);
        glBegin(GL_LINES);
        glVertex3f(Ow.at<float>(0) * mCoordinateScale,
                   Ow.at<float>(1) * mCoordinateScale,
                   Ow.at<float>(2) * mCoordinateScale);
        glVertex3f(Owp.at<float>(0) * mCoordinateScale,
                   Owp.at<float>(1) * mCoordinateScale,
                   Owp.at<float>(2) * mCoordinateScale);
        glEnd();
      }

      // Loops
      // step 3.3 连接闭环时形成的连接关系
      set<KeyFrame*> sLoopKFs = pKF->GetLoopEdges();
      if (!sLoopKFs.empty())
      {
        glLineWidth(2*mGraphLineWidth);
        glColor4f(0.f, 0.f, 1.f, 1.f);
        glBegin(GL_LINES);
        for (auto loopKF: sLoopKFs)
        {
          if (loopKF->mnId < pKF->mnId)
            continue;
          cv::Mat Owl = loopKF->GetCameraCenter();
          glVertex3f(Ow.at<float>(0) * mCoordinateScale,
                     Ow.at<float>(1) * mCoordinateScale,
                     Ow.at<float>(2) * mCoordinateScale);
          glVertex3f(Owl.at<float>(0) * mCoordinateScale,
                     Owl.at<float>(1) * mCoordinateScale,
                     Owl.at<float>(2) * mCoordinateScale);
        }
        glEnd();
      }
    }// 遍历完所有的关键帧
  }// 绘制共视图
}


void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
  //相机模型大小：宽度占总宽度比例为0.08
  const float &w = mCameraSize;
  const float h = w*0.75;
  const float z = w*0.6;

  glPushMatrix();
  //将4*4的矩阵Twc.m右乘一个当前矩阵
  //（由于使用了glPushMatrix函数，因此当前帧矩阵为世界坐标系下的单位矩阵）
  //因为OpenGL中的矩阵为列优先存储，因此实际为Tcw，即相机在世界坐标下的位姿
  //一个是整型,一个是浮点数类型
#ifdef HAVE_GLES
  glMultMatrixf(Twc.m);
#else
  glMultMatrixd(Twc.m);
#endif

  //设置绘制图形时线的宽度
  glLineWidth(mCameraLineWidth);
  //设置当前颜色为绿色(相机图标显示为绿色)
  glColor3f(0.0f,1.0f,0.0f);
  //用线将下面的顶点两两相连
  glBegin(GL_LINES);
  glVertex3f(0,0,0);
  glVertex3f(w,h,z);
  glVertex3f(0,0,0);
  glVertex3f(w,-h,z);
  glVertex3f(0,0,0);
  glVertex3f(-w,-h,z);
  glVertex3f(0,0,0);
  glVertex3f(-w,h,z);

  glVertex3f(w,h,z);
  glVertex3f(w,-h,z);

  glVertex3f(-w,h,z);
  glVertex3f(-w,-h,z);

  glVertex3f(-w,h,z);
  glVertex3f(w,h,z);

  glVertex3f(-w,-h,z);
  glVertex3f(w,-h,z);
  glEnd();

  glPopMatrix();
}

//设置当前帧相机的位姿, 设置这个函数是因为要处理多线程的操作
void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
{
  unique_lock<mutex> lock(mMutexCamera);
  mCameraPose = Tcw.clone();
}

// 将相机位姿mCameraPose由Mat类型转化为OpenGlMatrix类型
void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
  if(!mCameraPose.empty())
  {
    cv::Mat Rwc(3,3,CV_32F);
    cv::Mat twc(3,1,CV_32F);
    {
      unique_lock<mutex> lock(mMutexCamera);
      Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
      twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
    }

    M.m[0] = Rwc.at<float>(0,0);
    M.m[1] = Rwc.at<float>(1,0);
    M.m[2] = Rwc.at<float>(2,0);
    M.m[3]  = 0.0;

    M.m[4] = Rwc.at<float>(0,1);
    M.m[5] = Rwc.at<float>(1,1);
    M.m[6] = Rwc.at<float>(2,1);
    M.m[7]  = 0.0;

    M.m[8] = Rwc.at<float>(0,2);
    M.m[9] = Rwc.at<float>(1,2);
    M.m[10] = Rwc.at<float>(2,2);
    M.m[11]  = 0.0;

    M.m[12] = twc.at<float>(0) * mCoordinateScale;
    M.m[13] = twc.at<float>(1) * mCoordinateScale;
    M.m[14] = twc.at<float>(2) * mCoordinateScale;
    M.m[15]  = 1.0;
  }
  else
    M.SetIdentity();
}

} //namespace ORB_SLAM