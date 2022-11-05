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


#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <vector>

#include <opencv2/opencv.hpp>


namespace ORB_SLAM2
{

// THIS IS THE INITIALIZER FOR MONOCULAR SLAM. NOT USED IN THE STEREO OR RGBD CASE.
class Initializer
{
    typedef std::pair<int,int> Match;
public:
    /**
     * @brief 构造函数
     *
     * @param[in] sigma 重投影误差阈值
     */
    explicit Initializer(float sigma = 1.0);

    /**
     * @brief 通过两帧三角化地图点
     *
     * @param[in] ReferFrame            参考帧，通常为初始帧
     * @param[in] CurrentFrame          当前帧，也就是SLAM意义上的第二帧
     * @param[in] vMatches12            当前帧（2）和参考帧（1）图像中特征点的匹配关系
     *                                  vMatches12[i]解释：i表示帧1中关键点的索引值，vMatches12[i]的值为帧2的关键点索引值
     *                                  没有匹配关系的话，vMatches12[i]值为 -1
     * @param[in & out] vP3D            三角化测量之后的三维地图点
     * @param[in & out] vbTriangulated  标记三角化点是否有效，有效为true
     * @return true                     该帧可以成功初始化，返回true
     * @return false                    该帧不满足初始化条件，返回false
     */
    bool initializeMapPoints(const Frame &ReferFrame,
                             const Frame &CurrentFrame,
                             const std::vector<int> &vMatches12,
                             std::vector<cv::Point3f> &vP3D,
                             std::vector<bool> &vbTriangulated);
	
private:
    /**
     * @brief 三角化地图点
     * @param[in]   poseTcw1    参考帧的位姿 4x4Tcw矩阵
     * @param[in]   poseTcw2    当前帧的位姿 4x4Tcw矩阵
     * @param[in]   vKeys1			参考帧特征点
     * @param[in]   vKeys2			当前帧特征点
     * @param[in]   vMatches12	两帧特征点的匹配关系
     * @param[in]   K           相机内参矩阵
     * @param[in]   th2				  重投影误差的阈值的平方
     * @param[out]  vP3D				三角化测量之后的特征点的空间坐标
     * @param[out]  vbGood			特征点（对）中是good点的标记
     * @param[out]  parallax		特征点对中较小的视差角度°（不是最小视差角度）
     * @return	int 返回本组解中good点的数目
     */
    static int triangulatePoints(const cv::Mat &poseTcw1,
                          const cv::Mat &poseTcw2,
                          const std::vector<cv::KeyPoint> &vKeys1,
                          const std::vector<cv::KeyPoint> &vKeys2,
                          const std::vector<Match> &vMatches12,
                          const cv::Mat &K,
                          float th2,
                          std::vector<cv::Point3f> &vP3D,
                          std::vector<bool> &vbGood,
                          float &parallax);


    /**
     * @brief 给定投影矩阵P1,P2和图像上的匹配特征点点kp1,kp2，从而计算三维点坐标
     * 
     * @param[in] kp1               特征点, in reference frame
     * @param[in] kp2               特征点, in current frame
     * @param[in] P1                投影矩阵P1
     * @param[in] P2                投影矩阵P2
     * @param[in & out] x3D         计算的三维点
     */
    static void triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);


    // Standard Deviation and Variance
    float mSigma, mSigma2; // 重投影误差阈值
};

} //namespace ORB_SLAM

#endif // INITIALIZER_H
