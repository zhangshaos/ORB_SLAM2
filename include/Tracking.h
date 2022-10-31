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


#ifndef TRACKING_H
#define TRACKING_H

#include <mutex>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/imgproc/types_c.h>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"


namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

/**
 * @brief  追踪当前帧功能
 * 
 */
class Tracking
{
public:
    /**
     * @brief 构造函数
     * 
     * @param[in] pSys              系统实例 
     * @param[in] pVoc              字典指针
     * @param[in] pFrameDrawer      帧绘制器
     * @param[in] pMapDrawer        地图绘制器
     * @param[in] pMap              地图句柄
     * @param[in] pKFDB             关键帧数据库句柄
     * @param[in] strSettingPath    配置文件路径
     * @param[in] sensor            传感器类型
     */
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor);

    /**
     * @brief 处理单目输入图像
     * 
     * @param[in] im            图像
     * @param[in] timestamp     时间戳
     * @return cv::Mat          世界坐标系到该帧相机坐标系的变换矩阵
     */
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);

    /**
     * @brief 设置局部地图句柄
     * 
     * @param[in] pLocalMapper 局部建图器
     */
    void SetLocalMapper(LocalMapping* pLocalMapper);

    /**
     * @brief 设置回环检测器句柄
     * 
     * @param[in] pLoopClosing 回环检测器
     */
    void SetLoopClosing(LoopClosing* pLoopClosing);

    /**
     * @brief 设置可视化查看器句柄
     * 
     * @param[in] pViewer 可视化查看器
     */
    void SetViewer(Viewer* pViewer);

    /**
     * @brief 整个系统进行复位操作
     */
    void Reset();

public:

    // Tracking states
    enum eTrackingState
    {
        SYSTEM_NOT_READY = -1, //系统没有准备好的状态,一般就是在启动后加载配置文件和词典文件时候的状态
        NO_IMAGES_YET    = 0,  //当前无图像
        NOT_INITIALIZED  = 1,  //有图像但是没有完成初始化
        OK               = 2,  //正常时候的工作状态
        LOST             = 3   //系统已经跟丢了的状态
    };

    // 跟踪状态
    eTrackingState mState;
    eTrackingState mLastProcessedState; // 上一帧的跟踪状态.这个变量在绘制当前帧的时候会被使用到

    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImGray; // 还有当前帧的灰度图像? 提问,那么在双目输入和在RGBD输入的时候呢? 在双目输入和在RGBD输入时，为左侧图像的灰度图

    // Initialization Variables (Monocular)
    // - 之前的匹配
    std::vector<int> mvIniLastMatches;
    // - 初始化阶段中,当前帧中的特征点和参考帧中的特征点的匹配关系
    std::vector<int> mvIniMatches;// 跟踪初始化时前两帧之间的匹配
    // - 在初始化的过程中,保存参考帧中的特征点
    std::vector<cv::Point2f> mvbPrevMatched;
    // - 初始化过程中匹配后进行三角化得到的空间点
    std::vector<cv::Point3f> mvIniP3D;
    // - 初始化过程中的参考帧
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses; // 所有的参考关键帧的位姿; 看上面注释的意思,这里存储的也是相对位姿
    list<KeyFrame*> mlpReferences;      // 参考关键帧
    list<double> mlFrameTimes;          // 所有帧的时间戳? 还是关键帧的时间戳?
    list<bool> mlbLost;                 // 是否跟丢的标志

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for monocular
    void MonocularInitialization();
   
    // 单目输入的时候生成初始地图
    void CreateInitialMapMonocular();

    /**
     * @brief 检查上一帧中的MapPoints是否被替换
     * 
     * Local Mapping 线程可能会将关键帧中某些 MapPoints 进行替换，由于 Tracking 中需要用到 mLastFrame，这里检查并更新上一帧中被替换的 MapPoints
     * @see LocalMapping::SearchInNeighbors()
     */
    void CheckReplacedInLastFrame();

    /**
     * @brief 对参考关键帧的MapPoints进行跟踪
     * 
     * 1. 计算当前帧的词包，将当前帧的特征点分到特定层的nodes上
     * 2. 对属于同一node的描述子进行匹配
     * 3. 根据匹配对估计当前帧的姿态
     * 4. 根据姿态剔除误匹配
     * @return 如果匹配数大于10，返回true
     */
    bool TrackReferenceKeyFrame();

    /**
     * @brief 双目或rgbd摄像头根据深度值为上一帧产生新的MapPoints
     *
     * 在双目和rgbd情况下，选取一些深度小一些的点（可靠一些）
     * 可以通过深度值产生一些新的MapPoints
     */
    void UpdateLastFrame();
    
    /**
     * @brief 根据匀速度模型对上一帧的MapPoints进行跟踪
     * 
     * 1. 非单目情况，需要对上一帧产生一些新的MapPoints（临时）     
     * 2. 将上一帧的MapPoints投影到当前帧的图像平面上，在投影的位置进行区域匹配
     * 3. 根据匹配对估计当前帧的姿态
     * 4. 根据姿态剔除误匹配
     * @return 如果匹配数大于10，返回true
     * @see V-B Initial Pose Estimation From Previous Frame
     */
    bool TrackWithMotionModel();

    // 重定位模块
    bool Relocalization();

    /**
     * @brief 更新局部地图 LocalMap
     *
     * 局部地图包括：共视关键帧、临近关键帧及其子父关键帧，由这些关键帧观测到的 MapPoints
     */
    void UpdateLocalMap();
    
    /**
     * @brief 更新局部地图点（来自局部关键帧）
     * 
     */
    void UpdateLocalPoints();

   /**
     * @brief 更新局部关键帧
     * 方法是遍历当前帧的MapPoints，将观测到这些MapPoints的关键帧和相邻的关键帧及其父子关键帧，作为mvpLocalKeyFrames
     * Step 1：遍历当前帧的MapPoints，记录所有能观测到当前帧MapPoints的关键帧 
     * Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有三个策略
     * Step 2.1 策略1：能观测到当前帧MapPoints的关键帧作为局部关键帧 （将邻居拉拢入伙）
     * Step 2.2 策略2：遍历策略1得到的局部关键帧里共视程度很高的关键帧，将他们的家人和邻居作为局部关键帧
     * Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
     */
    void UpdateLocalKeyFrames();

    /**
     * @brief 对Local Map的MapPoints进行跟踪
     * Step 1：更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
     * Step 2：在局部地图中查找与当前帧匹配的MapPoints, 其实也就是对局部地图点进行跟踪
     * Step 3：更新局部所有MapPoints后对位姿再次优化
     * Step 4：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
     * Step 5：根据跟踪匹配数目及回环情况决定是否跟踪成功
     * @return true         跟踪成功
     * @return false        跟踪失败
     */
    bool TrackLocalMap();

    /**
     * @brief 对 Local MapPoints 进行跟踪
     * 
     * 在局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
     */
    void SearchLocalPoints();

    /**
     * @brief 断当前帧是否为关键帧
     * @return true if needed
     */
    bool NeedNewKeyFrame();

    /**
     * @brief 创建新的关键帧
     *
     * 对于非单目的情况，同时创建新的MapPoints
     */
    void CreateNewKeyFrame();

    // Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    // ORB
    // orb特征提取器，不管单目还是双目，mpORBextractorLeft 都要用到
    // 如果是双目，则要用到 mpORBextractorRight
    // 如果是单目，在初始化的时候使用 mpIniORBextractor 而不是 mpORBextractorLeft，
    // mpIniORBextractor 属性中提取的特征点个数是 mpORBextractorLeft 的两倍

    // 作者自己编写和改良的ORB特征点提取器
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    // 在初始化的时候使用的特征点提取器,其提取到的特征点个数会更多
    ORBextractor* mpIniORBextractor;

    // BoW 词袋模型相关
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    Initializer* mpInitializer;

    // Local Map 局部地图相关=
    KeyFrame* mpReferenceKF;// 当前关键帧就是参考帧
    // 局部关键帧集合
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    // 局部地图点的集合
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    System* mpSystem;
    
    // Drawers  可视化查看器相关
    Viewer* mpViewer;
    // 帧绘制器句柄
    FrameDrawer* mpFrameDrawer;
    // 地图绘制器句柄
    MapDrawer* mpMapDrawer;

    // Map
    // (全局)地图句柄
    Map* mpMap;

    // Calibration matrix  相机的参数矩阵相关
    cv::Mat mK;         // 相机的内参数矩阵
    cv::Mat mDistCoef;  // 相机的去畸变参数

    // New KeyFrame rules (according to fps)
    // 新建关键帧和重定位中用来判断最小最大时间间隔，和帧率有关
    int mMinFrames;
    int mMaxFrames;

    // Current matches in frame
    // 当前帧中的进行匹配的内点,将会被不同的函数反复使用
    int mnMatchesInliers;

    // Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    Frame mLastFrame;
    unsigned int mnLastKeyFrameId; // 上一个关键帧的ID
    unsigned int mnLastRelocFrameId; // 上一次重定位的那一帧的ID

    // Motion Model
    cv::Mat mVelocity;

    // Color order (true RGB, false BGR, ignored if grayscale)
    // RGB图像的颜色通道顺序
    bool mbRGB;
};  //class Tracking

} //namespace ORB_SLAM

#endif // TRACKING_H
