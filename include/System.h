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



#ifndef SYSTEM_H
#define SYSTEM_H

#include <string>
#include <thread>
#include <optional>
#include <opencv2/core/core.hpp>
#include <sophus/geometry.hpp>

#include "ORBVocabulary.h"


namespace ORB_SLAM2
{

// 要用到的其他类的前视声明
  class Viewer;
  class FrameDrawer;
  class MapPoint;
  class Map;
  class Tracking;
  class LocalMapping;
  class LoopClosing;
  class KeyFrameDatabase;
  class MapDrawer;


//本类的定义
  class System
  {
  public:
    System(const std::string &strVocFile,            //指定ORB字典文件的路径
           const std::string &strSettingsFile,       //指定配置文件的路径
           bool bUseViewer = true);       //指定是否使用可视化界面

    ~System();

    /**
     * 给定输入
     *
     * @param[in] im RGB|RGBA|Gray 图片
     * @param[in] timestamp 时间戳
     * @param[in] poseTcw 当前相机位姿（坐标系为相机坐标系：前z，右x，下y）
     * @param[in] imName 输入图像文件名
     * @return Tracking::State
     */
    int TrackMonocularWithPose(const cv::Mat &im, double timestamp,
                               const Sophus::SE3d& poseTcw,
                               const std::string& imName);

    // 获取从上次调用本函数后是否发生了比较大的地图变化
    bool MapChanged();

    // 复位 系统
    void Reset();

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    // 关闭系统，这将会关闭所有线程并且丢失曾经的各种数据
    void Shutdown();

    /**
     * 保存整个地图。
     *
     * @param filename[in]          保存文件路径（文件名）
     * @param revertTransform[in]   将ORB_SLAM2世界地图点转换为真实世界地图点，默认为空
     * @return
     */
    bool SaveMap(const std::string &filename, const Sophus::SE3d *revertTransform=nullptr);
    // bool LoadMap(const string &filename);

    // 获取最近的运动追踪状态、地图点追踪状态、特征点追踪状态
    int GetTrackingState();
    std::vector<MapPoint*> GetTrackedMapPoints();
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

    // 保存最近成功追踪到的地图点
    bool SaveTrackedMap(const std::string &filePath);

  protected:
    // 注意变量命名方式，类的变量有前缀m，如果这个变量是指针类型还要多加个前缀p，
    // 如果是线程那么加个前缀t

    // ORB vocabulary used for place recognition and feature matching.
    // 一个指针指向ORB字典
    ORBVocabulary* mpVocabulary{ nullptr };

    // KeyFrame database for place recognition (relocalization and loop detection).
    // 关键帧数据库的指针，这个数据库用于重定位和回环检测
    KeyFrameDatabase* mpKeyFrameDatabase{ nullptr };

    // 指向地图（数据库）的指针
    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    Map* mpMap{ nullptr };

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    // 追踪器，除了进行运动追踪外还要负责创建关键帧、创建新地图点和进行重定位的工作。详细信息还得看相关文件
    Tracking* mpTracker{ nullptr };

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    // 局部建图器。局部BA由它进行。
    LocalMapping* mpLocalMapper{ nullptr };

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    // 回环检测器，它会执行位姿图优化并且开一个新的线程进行全局BA
    LoopClosing* mpLoopCloser{ nullptr };

    // The viewer draws the map and the current camera pose. It uses Pangolin.
    // 查看器，可视化 界面
    Viewer* mpViewer{ nullptr };
    // 帧绘制器
    FrameDrawer* mpFrameDrawer{ nullptr };
    // 地图绘制器
    MapDrawer* mpMapDrawer{ nullptr };


    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    // 系统除了在主进程中进行运动追踪工作外，会创建局部建图线程、回环检测线程和查看器线程。
    std::thread mtLocalMapping;
    std::thread mtLoopClosing;
    std::thread mtViewer;

    // Reset flag
    bool mbReset{ false };

    // Tracking state
    // 追踪状态标志，注意前三个的类型和上面的函数类型相互对应
    int mTrackingState;
    std::vector<MapPoint*> mTrackedMapPoints;
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;

    // 存储当前系统输入位姿
    Sophus::SE3d mInPoseTcw{ Sophus::Matrix4d::Identity() };
    // 存储当前系统输入图片
    cv::Mat mInImage;
    // 存储当前输入图片文件名
    std::string mInImageName;
  };

}// namespace ORB_SLAM

#endif // SYSTEM_H
