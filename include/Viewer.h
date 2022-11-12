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


#ifndef VIEWER_H
#define VIEWER_H

#include <mutex>


namespace ORB_SLAM2
{

class Tracking;
class FrameDrawer;
class MapDrawer;
class System;

class Viewer
{
public:
  /**
   * @brief 构造函数
   *
   * @param[in] pSystem           系统实例
   * @param[in] pFrameDrawer      帧绘制器
   * @param[in] pMapDrawer        地图绘制器
   * @param[in] pTracking         追踪线程
   * @param[in] strSettingPath    设置文件的路径
   */
  Viewer(System* pSystem, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Tracking *pTracking, const string &strSettingPath);

  /**
   * @brief 进程的主函数, NOTICE 注意到这里提到了它是根据相机图像的更新帧率来绘制图像的
   * @detials Main thread function. Draw points, keyframes, the current camera pose and the last processed
   *  frame. Drawing is refreshed according to the camera fps. We use Pangolin.
   */
  void Run();

  /**
   * @brief 请求停止当前进程
   *
   */
  void RequestFinish();

  /**
   * @brief 请求重置更新图像数据，即暂停渲染
   *
   */
  void RequestReset();

  /**
   * @brief 请求恢复渲染，从暂停（重置）中恢复出来
   *
   */
  void RequestResetOver();

  /**
   * @brief 当前是否有停止当前进程的请求
   *
   */
  bool isFinished();

  // UI窗口的最大高度和宽度
  static const int MaxViewerHeight, MaxViewerWidth;

private:
  //系统对象指针
  System* mpSystem;
  //帧绘制器
  FrameDrawer* mpFrameDrawer;
  //地图绘制器
  MapDrawer* mpMapDrawer;
  //追踪线程句柄
  Tracking* mpTracker;

  // 1/fps in ms
  // 每一帧图像持续的时间
  double mT;
  // 图像的尺寸
  float mImageWidth, mImageHeight;
  // 显示窗口的的查看视角
  float mViewpointX, mViewpointY, mViewpointZ;
  // 相机的焦距
  float mCameraFocal;
  // 坐标系放缩，太大的坐标系在屏幕上显示不出来
  float mCoordinateScale;
  // 记录上一帧渲染花费的时间（毫秒）
  unsigned long long lastRenderingMilliseconds;

  // 请求结束当前线程的标志
  bool mbFinishRequested;
  // 当前线程是否已经终止
  bool mbFinished;
  // 是否有重置请求
  bool mbResetRequested;
};

} // Namespace ORB_SLAM2

#endif // VIEWER_H
	

