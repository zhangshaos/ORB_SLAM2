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

#include "Frame.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Tracking.h"
#include "System.h"
#include "Viewer.h"


namespace ORB_SLAM2
{

const int Viewer::MaxViewerHeight = 768,
          Viewer::MaxViewerWidth = 1024;

//查看器的构造函数
Viewer::Viewer(System* pSystem,
               FrameDrawer *pFrameDrawer,
               MapDrawer *pMapDrawer,
               Tracking *pTracking,
               const string &strSettingPath):
  mpSystem(pSystem),
  mpFrameDrawer(pFrameDrawer),
  mpMapDrawer(pMapDrawer),
  mpTracker(pTracking),
  mbFinishRequested(false),
  mbFinished(true),
  mbResetRequested(false)
{
  //从文件中读取相机的帧频
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

  float fps = fSettings["Camera.fps"];
  if(fps<1)
    fps=30;
  //计算出每一帧所持续的时间
  mT = 1e3/fps;

  //从配置文件中获取图像的长宽参数
  mImageWidth = fSettings["Camera.width"];
  mImageHeight = fSettings["Camera.height"];
  if(mImageWidth<1 || mImageHeight<1)
  {
    //默认值
    mImageWidth = 640;
    mImageHeight = 480;
  }

  //读取视角
  mViewpointX = fSettings["Viewer.ViewpointX"];
  mViewpointY = fSettings["Viewer.ViewpointY"];
  mViewpointZ = fSettings["Viewer.ViewpointZ"];
  mCameraFocal = fSettings["Viewer.CameraFocal"];
  mCoordinateScale = fSettings["Viewer.CoordinateScale"];
}

// pangolin库的文档：http://docs.ros.org/fuerte/api/pangolin_wrapper/html/namespacepangolin.html
// 查看器的主进程看来是外部函数所调用的
void Viewer::Run()
{
  // 这个变量配合SetFinish函数用于指示该函数是否执行完毕
  mbFinished = false;

  pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer",
                                Viewer::MaxViewerWidth,
                                Viewer::MaxViewerHeight);

  // 3D Mouse handler requires depth testing to be enabled
  // 启动深度测试，OpenGL只绘制最前面的一层，绘制时检查当前像素前面是否有别的像素，如果别的像素挡住了它，那它就不会绘制
  glEnable(GL_DEPTH_TEST);
  // Issue specific OpenGl we might need
  // 在OpenGL中使用颜色混合
  glEnable(GL_BLEND);
  // 选择混合选项
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // 新建按钮和选择框，第一个参数为按钮的名字，第二个为默认状态，第三个为是否有选择框
  pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
  pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
  pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
  pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
  pangolin::Var<bool> menuShowGraph("menu.Show Graph",true,true);
  pangolin::Var<bool> menuReset("menu.Reset",false,false);

  // Define Camera Render Object (for view / scene browsing)
  // 定义相机投影模型：ProjectionMatrix(w, h, fu, fv, u0, v0, zNear, zFar)
  // 定义观测方位向量：观测点位置：(mViewpointX mViewpointY mViewpointZ)
  //                观测目标位置：(0, 0, 0)
  //                观测的方位向量：(0.0,-1.0, 0.0)
  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(MaxViewerWidth, MaxViewerHeight,
                               mCameraFocal, mCameraFocal,
                               MaxViewerWidth/2.f, MaxViewerHeight/2.f,
                               1.f*mCoordinateScale, 100.f/mCoordinateScale),
    pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
  );

  // Add named OpenGL viewport to window and provide 3D Handler
  // 定义显示面板大小，ORB_SLAM中有左右两个面板，左边显示一些按钮，右边显示图形
  // 前两个参数（0.0, 1.0）表明宽度和面板纵向宽度和窗口大小相同
  // 中间两个参数（pangolin::Attach::Pix(175), 1.0）表明右边所有部分用于显示图形
  // 最后一个参数（-MaxViewerWidth/MaxViewerHeight）为显示长宽比
  pangolin::Handler3D handler3D(s_cam);
  pangolin::View& d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -(float)MaxViewerWidth/(float)MaxViewerHeight)
    .SetHandler(&handler3D);

  // 创建一个欧式变换矩阵,存储当前的相机位姿
  pangolin::OpenGlMatrix Twc;
  Twc.SetIdentity();

  // 创建当前帧图像查看器,谢晓佳在泡泡机器人的第35课中讲过这个;需要先声明窗口,再创建;否则就容易出现窗口无法刷新的情况
  cv::namedWindow("ORB-SLAM2: Frame Viewer");

  // ui设置
  bool bFollow = true;

  // 更新绘制的内容
  while (!mbFinishRequested)
  {
    while (mbResetRequested)
      // 等待其他线程重置状态，因为渲染线程依赖会访问其他线程的数据，
      // 如果其他线程在重置时被访问，可能访问不存在数据。
      this_thread::sleep_for(chrono::milliseconds(10));

    // 记录渲染耗时
    auto t0 = chrono::steady_clock::now();

    // 清除缓冲区中的当前可写的颜色缓冲 和 深度缓冲
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // step1：得到最新的相机位姿
    mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

    // step2：根据相机的位姿调整视角
    // menuFollowCamera为按钮的状态，bFollow为真实的状态
    if(menuFollowCamera && bFollow)
    {
      // 当之前也在跟踪相机时
      s_cam.Follow(Twc);
    }
    else if(menuFollowCamera && !bFollow)
    {
      // 当之前没有在跟踪相机时
      s_cam.SetModelViewMatrix(
        pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));   //? 不知道这个视角设置的具体作用和
      s_cam.Follow(Twc);
      bFollow = true;
    }
    else if(!menuFollowCamera && bFollow)
    {
      // 之前跟踪相机,但是现在菜单命令不要跟踪相机时
      bFollow = false;
    }

    d_cam.Activate(s_cam);
    // step 3：绘制地图和图像(3D部分)
    // 设置为白色，glClearColor(red, green, blue, alpha），数值范围(0, 1)

    // todo: 可视化信息更清晰

    glClearColor(1.0f,1.0f,1.0f,1.0f);
    // 绘制当前相机
    mpMapDrawer->DrawCurrentCamera(Twc);
    // 绘制关键帧和共视图
    if(menuShowKeyFrames || menuShowGraph)
      mpMapDrawer->DrawKeyFrames(menuShowKeyFrames,menuShowGraph);
    // 绘制地图点
    if(menuShowPoints)
      mpMapDrawer->DrawMapPoints();

    pangolin::FinishFrame();

    // step 4:绘制当前帧图像和特征点提取匹配结果
    cv::Mat im = mpFrameDrawer->DrawFrame();
    cv::imshow("ORB-SLAM2: Frame Viewer", im);
    // NOTICE 注意对于我所遇到的问题,ORB-SLAM2是这样子来处理的
    cv::waitKey(mT);

    // step 5 相应其他请求
    // 复位按钮
    if(menuReset)
    {
      // 将所有的GUI控件恢复初始状态
      menuShowGraph = true;
      menuShowKeyFrames = true;
      menuShowPoints = true;
      // 相关变量也恢复到初始状态
      bFollow = true;
      menuFollowCamera = true;
      // 告知系统复位
      mpSystem->Reset();
      // 按钮本身状态复位
      menuReset = false;
    }

    // 计算渲染耗时
    auto t1 = chrono::steady_clock::now();
    lastRenderingMilliseconds =
      chrono::duration_cast<chrono::milliseconds>(t1-t0).count();
  }

  // 销毁之前创建的窗口
  cv::destroyAllWindows();

  // 执行完成退出这个函数后,查看器进程就已经被销毁了
  mbFinished = true;
}

//外部函数调用,用来请求当前进程结束
void Viewer::RequestFinish()
{
  mbFinishRequested = true;
}

//判断当前进程是否已经结束
bool Viewer::isFinished()
{
  return mbFinished;
}

//外部线程请求当前查看器停止更新
void Viewer::RequestReset()
{
  mbResetRequested = true;
  this_thread::sleep_for(chrono::milliseconds(3*lastRenderingMilliseconds));
}

void Viewer::RequestResetOver()
{
  mbResetRequested = false;
}

} // Namespace ORB_SLAM2