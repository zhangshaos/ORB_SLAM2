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


#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include <mutex>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>


namespace ORB_SLAM2
{


/**
 * 直接对图片进行放缩，往往会导致摩尔纹，该函数内部进行多次放缩，以减小摩尔纹效应。
 * @param[in] im            cv::Mat 输入图片
 * @param[in] h             目标图片高度
 * @param[in] w             目标图片宽度
 * @param[in] scaleTimes    放缩次数，默认为3
 * @return cv::Mat 返回目标图片
 */
cv::Mat ResizeWithoutMoirePattern(const cv::Mat& im, int h, int w, int scaleTimes=3);


class Tracking;
class Viewer;
class Map;


/**
 * @brief 绘制每一帧图片、追踪的特征点等
 *
 */
class FrameDrawer
{
public:
  /**
   * @brief 构造函数
   *
   * @param[in] pMap  地图指针
   */
  explicit FrameDrawer(Map* pMap);

  // Update info from the last processed frame.
  /**
   * @brief 将跟踪线程的数据拷贝到绘图线程（图像、特征点、地图、跟踪状态）
   *
   * @param[in] pTracker 追踪线程
   */
  void Update(Tracking *pTracker);

  // Draw last processed frame.
  //
  /**
   * @brief 绘制最近处理过的帧,这个将会在可视化查看器的窗口中被创建
   *
   * @return cv::Mat 返回绘制完成的图像,可以直接进行显示
   */
  cv::Mat DrawFrame();

protected:

  /**
   * @brief 绘制底部的信息栏
   *
   * @param[in]  im           原始图像
   * @param[in]  nState       当前系统的工作状态
   * @param[out] imText       叠加后的图像
   */
  void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

  //当前绘制的图像
  cv::Mat mIm;
  //当前帧中特征点的数目
  int N;
  //当前帧中的特征点
  std::vector<cv::KeyPoint> mvCurrentKeys;
  //当前帧中特征点对应的地图点是否在地图中的标记
  //当前帧的特征点在地图中是否出现;后者是表示地图中没有出现,但是在当前帧中是第一次被观测得到的点
  std::vector<bool> mvbKeyPtInMap, mvbVO;
  //当前帧中追踪到的特征点计数
  int mnTracked, mnTrackedVO;
  //参考帧中的特征点
  std::vector<cv::KeyPoint> mvIniKeys;
  //当前帧特征点和参考帧特征点的匹配关系
  std::vector<int> mvIniMatches;
  //当前SLAM系统的工作状态
  int mState;

  //地图指针
  Map* mpMap;

  //线程锁
  std::mutex mMutex;
};

} //namespace ORB_SLAM

#endif // FRAMEDRAWER_H
