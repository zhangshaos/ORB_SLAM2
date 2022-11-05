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
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/imgproc/types_c.h>

#include "Frame.h"


namespace ORB_SLAM2
{

class ORBextractor;
class Viewer;
class FrameDrawer;
class MapDrawer;
class Map;
class Initializer;
class LocalMapping;
class LoopClosing;
class KeyFrameDatabase;
class System;

/**
 * @brief  追踪当前帧功能
 * 
 */
class Tracking
{
public:
    enum State
    {
      SYSTEM_NOT_READY = -1, //系统没有准备好的状态,一般就是在启动后加载配置文件和词典文件时候的状态
      NO_IMAGES_YET    = 0,  //当前无图像
      NOT_INITIALIZED  = 1,  //有图像但是没有完成初始化
      OK               = 2,  //正常时候的工作状态
      LOST             = 3   //系统已经跟丢了的状态
    };

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
     */
    Tracking(System* pSys,
             ORBVocabulary* pVoc,
             FrameDrawer* pFrameDrawer,
             MapDrawer* pMapDrawer,
             Map* pMap,
             KeyFrameDatabase* pKFDB,
             const std::string &strSettingPath);

    /**
     * @brief 处理单目输入图像
     *
     * @param[in] im            图像
     * @param[in] timestamp     时间戳
     * @param[in] poseTcw       当前位置和朝向
     * @return Tracking::State  追踪状态
     */
     Tracking::State trackImageWithPose(const cv::Mat &im,
                                        double timestamp,
                                        const cv::Mat &poseTcw);

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
    // 跟踪状态
    State mState;
    State mLastProcessedState; // 上一帧的跟踪状态. 这个变量在绘制当前帧的时候会被使用到

    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImGray;

    // Initialization Variables (Monocular)
    Frame mInitialFrame; // 初始化过程中的参考帧
    std::vector<int> mvInitialMatches; // 初始化阶段中,当前帧中的特征点和参考帧中的特征点的匹配关系
protected:
   /**
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
    void Initialization();
   
    // 生成初始地图，由 Initialization() 调用
    void CreateInitialMap(const std::vector<cv::Point3f>& vIniP3D);

    /**
     * @brief 检查上一帧中的MapPoints是否被替换
     *
     * 很多修改 lastFrame 的函数，应该都是为了可视化
     *
     * Local Mapping 线程可能会将关键帧中某些 MapPoints 进行替换，由于 Tracking 中需要用到 mLastFrame，这里检查并更新上一帧中被替换的 MapPoints
     * @see LocalMapping::FusePointsInNeighbors()
     */
    void CheckReplacedMapPointsInLastFrame();

    /**
     * 将当前帧已匹配的地图点重投影到图片上，计算误差error，
     * 并将error^2 > th2的地图点取消与特征点的匹配。
     *
     * 由 TrackWithInitialPose()、TrackWithReferenceKF() 调用
     *
     * @return 错误匹配(error^2 > th2)的数量
     */
    int CheckMatchesByProjection(float th2);

    /**
     * 尝试使用当前帧初始化的位姿来进行当前帧特征-过去帧地图点匹配
     *
     * @return is a successful tracking.
     */
    bool TrackWithInitialPose();

    /**
     * 尝试计算 Tracking 的参考KF地图点-当前帧特征匹配
     *
     * @return is a successful tracking.
     */
    bool TrackWithReferenceKF();

    
    /**
     * @brief 更新局部地图点（来自局部关键帧）
     * 这里的地图点不止当前帧可以看见的地图点
     *
     * 由 TrackLocalMap() 调用
     */
    void UpdateLocalMapPoints();

   /**
     * @brief 更新局部关键帧
     * 方法是遍历当前帧的MapPoints，将观测到这些MapPoints的关键帧和相邻的关键帧及其父子关键帧，作为mvpLocalKeyFrames
     * Step 1：遍历当前帧的MapPoints，记录所有能观测到当前帧MapPoints的关键帧 
     * Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有三个策略
     * Step 2.1 策略1：能观测到当前帧MapPoints的关键帧作为局部关键帧 （将邻居拉拢入伙）
     * Step 2.2 策略2：遍历策略1得到的局部关键帧里共视程度很高的关键帧，将他们的家人和邻居作为局部关键帧
     * Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
     *
     * 由 TrackLocalMap() 调用
     */
    void UpdateLocalKeyFrames();

    /**
     * @brief 用局部地图进行跟踪，进一步优化位姿
     *
     * 1. 更新局部地图，包括局部关键帧和关键点，
     * 局部地图包括：K1个关键帧、K2个临近关键帧和参考关键帧 + 由这些关键帧观测到的MapPoints
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
    bool TrackLocalMap();

    /**
     * @brief 对 Local MapPoints 进行跟踪
     * 
     * 在局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
     *
     * 由 TrackLocalMap() 调用
     * @return 返回新增匹配的地图点数量
     */
    int ProjectLocalPointsToCurrentFrame();

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

    // ORB
    // orb特征提取器，不管单目还是双目，mpORBextractorLeft 都要用到
    // 如果是双目，则要用到 mpORBextractorRight
    // 如果是单目，在初始化的时候使用 mpIniORBextractor 而不是 mpORBextractorLeft，
    // mpIniORBextractor 属性中提取的特征点个数是 mpORBextractorLeft 的两倍

    // 作者自己编写和改良的ORB特征点提取器
    ORBextractor* mpORBextractorLeft;
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

    // Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;
    // System
    System* mpSystem;
    // Drawers  可视化查看器相关
    Viewer* mpViewer;
    // 帧绘制器句柄
    FrameDrawer* mpFrameDrawer;
    // 地图绘制器句柄
    MapDrawer* mpMapDrawer;
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
    // 当前帧中和特征点匹配的地图点数量,将会被不同的函数反复使用
    int mnMatchesInliers;

    // Last Frame, KeyFrame and Relocalisation Info
    Frame mLastFrame;
    unsigned int mnLastKeyFrameId;   // 上一个关键帧的帧ID（不是关键帧ID）
    unsigned int mnLastRelocFrameId; // 上一次重定位的那一帧的帧ID

    // Color order (true RGB, false BGR, ignored if grayscale)
    // RGB图像的颜色通道顺序
    bool mbRGB;
};  //class Tracking

} //namespace ORB_SLAM

#endif // TRACKING_H
