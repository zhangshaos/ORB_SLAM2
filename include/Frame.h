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


#ifndef FRAME_H
#define FRAME_H

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include "MapPoint.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"


namespace ORB_SLAM2
{
	
/**
 * @name 定义一帧中有多少个图像网格
 * @{
 */

/**
 * @brief 网格的行数
 * 
 */
#define FRAME_GRID_ROWS 48
/**
 * @brief 网格的列数
 * 
 */
#define FRAME_GRID_COLS 64

/** @} */

class MapPoint;
class KeyFrame;

/**
 * @brief 帧
 */
class Frame
{
public:
	
    /**
     * @brief Construct a new Frame object without parameter. 
     * 
     */
    Frame();

    Frame(const Frame &frame);

    /**
     * @brief 为单目相机准备的帧构造函数
     * 
     * @param[in] imGray                            //灰度图
     * @param[in] timeStamp                         //时间戳
     * @param[in & out] extractor                   //ORB特征点提取器的句柄
     * @param[in] voc                               //ORB字典的句柄
     * @param[in] K                                 //相机的内参数矩阵
     * @param[in] distCoef                          //相机的去畸变参数
     */
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef);

    /**
     * @brief 提取图像的ORB特征，提取的关键点存放在mvKeys，描述子存放在mDescriptors
     *
     * @param[in] im            等待提取特征点的图像
     */
    void ExtractORB(const cv::Mat &im);

    /**
     * @brief 计算词袋模型, 存放在mBowVec中
     * @details 计算词包 mBowVec 和 mFeatVec ，其中 mFeatVec 记录了属于第i个node（在第4层）的ni个描述子
     * @see CreateInitialMapMonocular() TrackReferenceKeyFrame() Relocalization()
     */
    void ComputeBoW();

    /**
     * @brief 用 Tcw 更新 mTcw 以及类中存储的一系列位姿
     * 
     * @param[in] Tcw 从世界坐标系到当前帧相机位姿的变换矩阵
     */
    void SetPose(cv::Mat Tcw);

    /**
     * @brief 根据相机位姿,计算相机的旋转,平移和相机中心等矩阵.
     * @details 其实就是根据 Tcw 计算 mRcw、mtcw 和 mRwc、mOw.
     */
    void UpdatePoseMatrices();

    /**
     * @brief 返回位于当前帧位姿时,相机的中心
     * 
     * @return cv::Mat 相机中心在世界坐标系下的3D点坐标
     */
    inline cv::Mat GetCameraCenter()
	  {
        return mOw.clone();
    }

    /**
     * @brief Get the Rotation Inverse object
     * mRwc 存储的是从当前相机坐标系到世界坐标系所进行的旋转，而我们一般用的旋转则说的是从世界坐标系到当前相机坐标系的旋转
     * @return 返回从当前帧坐标系到世界坐标系的旋转
     */
    inline cv::Mat GetRotationInverse()
	  {
        return mRwc.clone();
    }

    /**
     * @brief 判断路标点是否在视野中，並且填充一些MapPoint用在 tracking 时的变量
     * 步骤
     * Step 1 获得这个地图点的世界坐标
     * Step 2 关卡一：检查这个地图点在当前帧的相机坐标系下，是否有正的深度.如果是负的，表示出错，返回false
     * Step 3 关卡二：将MapPoint投影到当前帧的像素坐标(u,v), 并判断是否在图像有效范围内
     * Step 4 关卡三：计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内
     * Step 5 关卡四：计算当前视角和“法线”夹角的余弦值, 若小于设定阈值，返回false
     * Step 6 根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
     * Step 7 记录计算得到的一些参数
     * @param[in] pMP                       当前地图点
     * @param[in] viewingCosLimit           夹角余弦，用于限制地图点和光心连线和法线的夹角
     * @return true                         地图点合格，且在视野内
     * @return false                        地图点不合格，抛弃
     */
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    /**
     * @brief 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，
     * 记录在 nGridPosX, nGridPosY 里，返回true，没找到返回false
     * 
     * @param[in] kp                    给定的特征点
     * @param[in & out] posX            特征点所在网格坐标的横坐标
     * @param[in & out] posY            特征点所在网格坐标的纵坐标
     * @return true                     如果找到特征点所在的网格坐标，返回true
     * @return false                    没找到返回false
     */
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    /**
     * @brief 找到在 以x,y为中心,半径为r的圆形内且金字塔层级在[minLevel, maxLevel]的特征点
     * 
     * @param[in] x                     特征点坐标x
     * @param[in] y                     特征点坐标y
     * @param[in] r                     搜索半径 
     * @param[in] minLevel              最小金字塔层级
     * @param[in] maxLevel              最大金字塔层级
     * @return vector<size_t>           返回搜索到的候选匹配点id
     */
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;

public:

    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary; // 用于重定位的ORB特征字典

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK; // 相机的内参数矩阵
	  // 注意这里的相机内参数其实都是类的静态成员变量；此外相机的内参数矩阵和矫正参数矩阵却是普通的成员变量。
    
    static float fx;        // x轴方向焦距
    static float fy;        // y轴方向焦距
    static float cx;        // x轴方向光心偏移
    static float cy;        // y轴方向光心偏移
    static float invfx;     // x轴方向焦距的逆
    static float invfy;     // x轴方向焦距的逆

    cv::Mat mDistCoef; // 去畸变参数

    // Number of KeyPoints.
    int N;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    // mvKeys: 原始左图像提取出的特征点（未校正）用来可视化
    // mvKeysRight: 原始右图像提取出的特征点（未校正）
    // mvKeysUn: 校正mvKeys后的特征点，对于双目摄像头，一般得到的图像都是校正好的，再校正一次有点多余

    std::vector<cv::KeyPoint> mvKeys;       // 原始左图像提取出的特征点（未校正）
    std::vector<cv::KeyPoint> mvKeysRight;  // 原始右图像提取出的特征点（未校正）
    std::vector<cv::KeyPoint> mvKeysUn;     // 校正mvKeys后的特征点
    // 之所以对于双目摄像头只保存左图像矫正后的特征点,是因为对于双目摄像头,一般得到的图像都是矫正好的,这里再矫正一次有些多余.\n
    // 校正操作是在帧的构造函数中进行的。
    
    // Bag of Words Vector structures.
    // 内部实际存储的是 std::map<WordId, WordValue>
    // WordId 和 WordValue 表示 Word 在叶子中的 id 和权重
    DBoW2::BowVector mBowVec;
    // 内部实际存储 std::map<NodeId, std::vector<unsigned int>>
    // NodeId 表示节点id，std::vector<unsigned int> 中实际存的是该节点id下所有特征点在图像中的索引
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    // 左目摄像头和右目摄像头特征点对应的描述子
    cv::Mat mDescriptors, mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.
    // 每个特征点对应的MapPoint.如果特征点没有对应的地图点,那么将存储一个空指针
    std::vector<MapPoint*> mvpMapPoints;

    // Flag to identify outlier associations.
    // 观测不到Map中的3D点
    // 属于外点的特征点标记,在 Optimizer::PoseOptimization 使用了
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
	  // 原来通过对图像分区域还能够降低重投影地图点时候的匹配复杂度
    // 注意到上面也是类的静态成员变量， 有一个专用的标志 mbInitialComputations 用来在帧的构造函数中标记这些静态成员变量是否需要被赋值
    // 坐标乘以 mfGridElementWidthInv 和 mfGridElementHeightInv 就可以确定在哪个格子
    static float mfGridElementWidthInv;
    // 坐标乘以 mfGridElementWidthInv 和 mfGridElementHeightInv 就可以确定在哪个格子
    static float mfGridElementHeightInv;
    

    // 每个格子分配的特征点数，将图像分成格子，保证提取的特征点比较均匀
    // FRAME_GRID_ROWS 48
    // FRAME_GRID_COLS 64
	  // 这个向量中存储的是每个图像网格内特征点的id（左图）
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];


    // Camera pose.
    cv::Mat mTcw; // 相机姿态，世界坐标系到相机坐标坐标系的变换矩阵,是我们常规理解中的相机位姿

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId; // Current Frame id.

    // Reference Keyframe.
    // 普通帧与自己共视程度最高的关键帧作为参考关键帧
    KeyFrame* mpReferenceKF;

    // Scale pyramid info.
    int mnScaleLevels;                  // 图像金字塔的层数
    float mfScaleFactor;                // 图像金字塔的尺度因子
    float mfLogScaleFactor;             // 图像金字塔的尺度因子的对数值，用于仿照特征点尺度预测地图点的尺度
                                  
    vector<float> mvScaleFactors;		    // 图像金字塔每一层的缩放因子
    vector<float> mvInvScaleFactors;	  // 以及上面的这个变量的倒数
    vector<float> mvLevelSigma2;		    // todo：目前在 frame.cpp 中没有用到
    vector<float> mvInvLevelSigma2;		  // 上面变量的倒数

    // Undistorted Image Bounds (computed once).
    /**
     * @name 用于确定画格子时的边界 
     * @note（未校正图像的边界，只需要计算一次，因为是类的静态成员变量）
     * @{
     */
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;


    /**
     * @brief 一个标志，标记是否已经进行了这些初始化计算
     * @note 由于第一帧以及SLAM系统进行重新校正后的第一帧会有一些特殊的初始化处理操作，所以这里设置了这个变量.
     * 如果这个标志被置位，说明再下一帧的帧构造函数中要进行这个“特殊的初始化操作”，如果没有被置位则不用。
    */ 
    static bool mbInitialComputations;

private:
    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    /**
     * @brief 用内参对特征点去畸变，结果报存在mvKeysUn中
     * 
     */
    void UndistortKeyPoints();

    /**
     * @brief 计算去畸变图像的边界
     * 
     * @param[in] imLeft 需要计算边界的图像
     */
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    /**
     * @brief 将提取到的特征点分配到图像网格中
     * @details 该函数由构造函数调用
     * 
     */
    void AssignFeaturesToGrid();

    // Rotation, translation and camera center
    cv::Mat mRcw; // Rotation from world to camera
    cv::Mat mtcw; // Translation from world to camera
    cv::Mat mRwc; // Rotation from camera to world
    cv::Mat mOw;  // mtwc,Translation from camera to world
};

}// namespace ORB_SLAM

#endif // FRAME_H
