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


#include "Thirdparty/DBoW2/DUtils/Random.h"
#include <glog/logging.h>

#include "Frame.h"
#include "MapPoint.h"
#include "Map.h"
#include "Optimizer.h"
#include "Initializer.h"


namespace ORB_SLAM2
{


/**
 * @brief 构造函数
 *
 * @param[in] sigma 重投影误差阈值
 */
Initializer::Initializer(float sigma) : mSigma(sigma), mSigma2(sigma*sigma)
{

}


/**
 * @brief 给定投影矩阵P1,P2和图像上的匹配特征点点kp1,kp2，从而计算三维点坐标
 *
 * @param[in] kp1               特征点, in reference frame
 * @param[in] kp2               特征点, in current frame
 * @param[in] P1                投影矩阵P1
 * @param[in] P2                投影矩阵P2
 * @param[in & out] x3D         计算的三维点
 */
void Initializer::triangulate(
  const cv::KeyPoint &kp1,    //特征点, in reference frame
  const cv::KeyPoint &kp2,    //特征点, in current frame
  const cv::Mat &P1,          //投影矩阵P1
  const cv::Mat &P2,          //投影矩阵P2
  cv::Mat &x3D)               //三维点
{
  // 原理
  // Trianularization: 已知匹配特征点对{x x'} 和 各自相机矩阵{P P'}, 估计三维点 X
  // x' = P'X  x = PX
  // 它们都属于 x = aPX模型
  //                         |X|
  // |x|     |p1 p2  p3  p4 ||Y|     |x|    |--p0--||.|
  // |y| = a |p5 p6  p7  p8 ||Z| ===>|y| = a|--p1--||X|
  // |z|     |p9 p10 p11 p12||1|     |z|    |--p2--||.|
  // 采用DLT的方法：x叉乘PX = 0
  // |yp2 -  p1|     |0|
  // |p0 -  xp2| X = |0|
  // |xp1 - yp0|     |0|
  // 两个点:
  // |yp2   -  p1  |     |0|
  // |p0    -  xp2 | X = |0| ===> AX = 0
  // |y'p2' -  p1' |     |0|
  // |p0'   - x'p2'|     |0|
  // 变成程序中的形式：
  // |xp2  - p0 |     |0|
  // |yp2  - p1 | X = |0| ===> AX = 0
  // |x'p2'- p0'|     |0|
  // |y'p2'- p1'|     |0|
  // 然后就组成了一个四元一次正定方程组，SVD求解，右奇异矩阵的最后一行就是最终的解.

  //这个就是上面注释中的矩阵A
  cv::Mat A(4,4,CV_32F);

  //构造参数矩阵A
  A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
  A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
  A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
  A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

  //奇异值分解的结果
  cv::Mat u,w,vt;
  //对系数矩阵A进行奇异值分解
  cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  //根据前面的结论，奇异值分解右矩阵的最后一行其实就是解，原理类似于前面的求最小二乘解，四个未知数四个方程正好正定
  //别忘了我们更习惯用列向量来表示一个点的空间坐标
  x3D = vt.row(3).t();
  //为了符合其次坐标的形式，使最后一维为1
  x3D = x3D.rowRange(0,3) / x3D.at<float>(3);
}


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
bool Initializer::initializeMapPoints(const Frame &ReferFrame,
                                      const Frame &CurrentFrame,
                                      const vector<int> &vMatches12,
                                      std::vector<cv::Point3f> &vP3D,
                                      std::vector<bool> &vbTriangulated)
{
  // Reference Frame: 1, Current Frame: 2
  std::vector<Match> matches12;
  for(size_t i=0, iend=vMatches12.size(); i<iend; ++i)
    if(vMatches12[i]>=0)
      matches12.emplace_back(i, vMatches12[i]);
  float parallax;
  int nGood = triangulatePoints(
    ReferFrame.getPose(), CurrentFrame.getPose(),
    ReferFrame.getKeyPoints(), CurrentFrame.getKeyPoints(),
    matches12,
    ReferFrame.getCameraIntrincs(),
    4.f * mSigma2,
    vP3D,
    vbTriangulated,
    parallax);

  const float minParallax = 1.0; // 视差角度 1°
  const int minTriangulated = 50;
  return parallax > minParallax && nGood >= minTriangulated;
}


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
int Initializer::triangulatePoints(const cv::Mat &poseTcw1,
                                   const cv::Mat &poseTcw2,
                                   const std::vector<cv::KeyPoint> &vKeys1,
                                   const std::vector<cv::KeyPoint> &vKeys2,
                                   const std::vector<Match> &vMatches12,
                                   const cv::Mat &K,
                                   float th2,
                                   std::vector<cv::Point3f> &vP3D,
                                   std::vector<bool> &vbGood,
                                   float &parallax)
{
  int N = vKeys1.size();
  vP3D.resize(N);
  vbGood.resize(N, false);

  LOG(INFO) << "Camera Intrincs matrix:\n" << K;

  // Step 1：计算相机的投影矩阵
  cv::Mat P1(3, 4, CV_32F);
  poseTcw1.rowRange(0,3).colRange(0,4).copyTo(P1);
  LOG(INFO) << "poseTcw1:\n" << P1;
  P1 = K * P1;
  cv::Mat R1 = poseTcw1.rowRange(0,3).colRange(0,3);
  cv::Mat t1 = poseTcw1.col(3).rowRange(0,3);
  cv::Mat O1 = -(R1.t() * t1); // 光心设置为世界坐标系下的原点
  LOG(INFO) << "O1:\n" << O1;

  cv::Mat P2(3, 4, CV_32F);
  poseTcw2.rowRange(0,3).colRange(0,4).copyTo(P2);
  LOG(INFO) << "PoseTcw2:\n" << P2;
  P2 = K * P2;
  cv::Mat R2 = poseTcw2.rowRange(0,3).colRange(0,3);
  cv::Mat t2 = poseTcw2.col(3).rowRange(0,3);
  cv::Mat O2 = -(R2.t() * t2);
  LOG(INFO) << "O2:\n" << O2;

  std::vector<float> vCosParallax; // 存储计算出来的每对特征点的视差
  vCosParallax.reserve(N);
  int nGood = 0;
  for(const auto & match12 : vMatches12)
  {
    // Step 2 获取特征点对，调用Triangulate() 函数进行三角化，得到三角化测量之后的3D点坐标
    const cv::KeyPoint &kp1 = vKeys1[match12.first];
    const cv::KeyPoint &kp2 = vKeys2[match12.second];
    cv::Mat p3d;
    triangulate(kp1, kp2, P1, P2, p3d); // 输出三角化测量之后特征点的空间坐标
    //LOG(INFO) << "p3d:\n" << p3d;
    cv::Mat p3dHomo(4, 1, CV_32F, cv::Scalar(1.f));
    p3d.copyTo(p3dHomo.rowRange(0, 3));

    // Step 3 第一关：检查三角化的三维点坐标是否合法（非无穷值）
    // 只要三角测量的结果中有一个是无穷大的就说明三角化失败，跳过对当前点的处理，进行下一对特征点的遍历
    if(!isfinite(p3d.at<float>(0)) || !isfinite(p3d.at<float>(1)) || !isfinite(p3d.at<float>(2)))
    {
      vbGood[match12.first]=false;
      continue;
    }

    // Step 4 第二关：通过三维点深度值正负、两相机光心视差角大小来检查是否合法
    cv::Mat normal1 = p3d - O1;
    float dist1 = cv::norm(normal1);
    cv::Mat normal2 = p3d - O2;
    float dist2 = cv::norm(normal2);
    float cosParallax = normal1.dot(normal2) / (dist1*dist2);
    cv::Mat zuv1 = P1 * p3dHomo; // z * (u,v,1)^t
    cv::Mat zuv2 = P2 * p3dHomo;
    //LOG(INFO) << "Camera p1:\n" << zuv1 << '\n'
    //          << "Camera p2:\n" << zuv2 << '\n'
    //          << "CosParallax: " << cosParallax;
    if(cosParallax >= 0.99998 || zuv1.at<float>(2) <= 0 || zuv2.at<float>(2) <= 0)
      // 如果视差较小（cos值大），或者深度值为负值，则排除
      // 这里0.99998 对应的角度为0.36°
      continue;

    // Step 5 第三关：计算空间点在参考帧和当前帧上的重投影误差，如果大于阈值则舍弃
    cv::Mat ptHomo1 = zuv1 / zuv1.at<float>(2); // (u,v,1)
    float im1x = ptHomo1.at<float>(0);
    float im1y = ptHomo1.at<float>(1);
    float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);
    //LOG(INFO) << "Reprojected point: (" << im1y << ", " << im1x << ")\n"
    //          << "Original point: (" << kp1.pt.y << ", " << kp1.pt.x << ")\n"
    //          << "Projection Error 1 ^2: " << squareError1 << '\n';
    if(squareError1 > th2)
      continue;

    cv::Mat ptHomo2 = zuv2 / zuv2.at<float>(2);
    float im2x = ptHomo2.at<float>(0);
    float im2y = ptHomo2.at<float>(1);
    float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);
    //LOG(INFO) << "Reprojected point: (" << im2y << ", " << im2x << ")\n"
    //          << "Original point: (" << kp2.pt.y << ", " << kp2.pt.x << ")\n"
    //          << "Projection Error 2 ^2: " << squareError2 << '\n';
    if(squareError2 > th2)
      continue;

    // Step 6 统计经过检验的3D点个数，记录3D点视差角
    vP3D[match12.first] = cv::Point3f(p3d.at<float>(0), p3d.at<float>(1), p3d.at<float>(2));
    vCosParallax.emplace_back(cosParallax);
    nGood++;
    // 不满足视差要求的点，会在Tracking::Initialization()中下一步
    // 重置 mvInitialMatches 对应项目为 -1，
    // 并且不会在 CreateInitialMap() 作为地图点。
    vbGood[match12.first] = true;
  }

  // Step 7 得到3D点中较小的视差角，并且转换成为角度制表示
  if(nGood > 0)
  {
    sort(vCosParallax.begin(), vCosParallax.end()); // 从小到大排序，注意vCosParallax值越大，视差越小

    // 排序后并没有取最小的视差角，而是取一个较小的视差角
    // 作者的做法：如果经过检验过后的有效3D点小于50个，那么就取最后那个最小的视差角(cos值最大)
    // 如果大于50个，就取排名第50个的较小的视差角即可，为了避免3D点太多时出现太小的视差角
    size_t idx = min(50, int(vCosParallax.size() - 1));
    parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
  }
  else
    parallax = 0;
  return nGood;
}


} //namespace ORB_SLAM