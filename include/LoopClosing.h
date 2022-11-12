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

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include <thread>
#include <mutex>
#include <set>
#include <map>
#include <list>
#include <vector>

#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"


namespace ORB_SLAM2
{

class Tracking;
class LocalMapping;
class KeyFrameDatabase;
class KeyFrame;

// 回环检测线程
class LoopClosing
{
public:
  // 相关的概念说明:
  // 组(group): 对于某个关键帧，其和其具有共视关系的关键帧组成了一个"组"。
  // 候选组(CandidateGroup): 对于某个候选回环关键帧，其和其具有共视关系的关键帧组成的一个"候选组"。
  // 连续(Consistent)：不同的候选组之间如果共同拥有一个及以上的关键帧，那么称这两个组之间具有连续关系。
  // 连续组(Consistent group): 具有连续性的多个候选组的集合。
  // 连续性(Consistency)：连续组中每个候选组的连续次数or长度，具体反映在数据类型 CandinateGroup.second上。

  // CandinateGroup.first对应每个“候选组”中的关键帧，CandinateGroup.second为每个“候选组”的"连续长度"
  typedef std::pair<std::set<KeyFrame*>,int> CandinateGroup;

  // 存储关键帧对象和位姿的键值对,这里是map的完整构造函数
  typedef std::map<KeyFrame*,                  //键
                   g2o::Sim3,                  //值
                   std::less<KeyFrame*>,       //排序算法
                   Eigen::aligned_allocator<std::pair<KeyFrame *const, g2o::Sim3> > // 指定分配器,和内存空间开辟有关. 为了能够使用Eigen库中的SSE和AVX指令集加速,需要将传统STL容器中的数据进行对齐处理
                  > KeyFrameAndPose;

public:

  /**
   * @brief 构造函数
   * @param[in] pMap          地图指针
   * @param[in] pDB           词袋数据库
   * @param[in] pVoc          词典
   * 3中的s需要被计算
   */
  LoopClosing(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc);

  /**
   * @brief 设置追踪线程的句柄
   * @param[in] pTracker 追踪线程的句柄
   */
  void SetTracker(Tracking* pTracker);

  /**
   * @brief 设置局部建图线程的句柄
   * @param[in] pLocalMapper
   */
  void SetLocalMapper(LocalMapping* pLocalMapper);

  /**
   * @brief 回环检测线程主函数
   */
  void Run();

  /**
   * @brief 将某个关键帧加入到回环检测的过程中,由局部建图线程调用
   * @param[in] pKF
   */
  void InsertKeyFrame(KeyFrame *pKF);

  /**
   * @brief 由外部线程调用,请求复位当前线程.在回环检测复位完成之前,该函数将一直保持堵塞状态
   */
  void RequestReset();

  /**
   * @brief 全局BA线程，这个函数将会再开一个线程运行
   * @param[in] nLoopKF 看名字是闭环关键帧,但是实际上给的是当前关键帧的ID
   */
  void RunGlobalBundleAdjustment(unsigned long nLoopKF);

  // 在回环纠正的时候调用,查看当前是否已经有一个全局优化的线程在进行
  bool isRunningGBA()
  {
    return mbRunningGBA;
  }

  /**
   * @brief 由外部线程调用,请求终止当前线程
   */
  void RequestFinish();

  /**
   * @brief 由外部线程调用,判断当前回环检测线程是否已经正确终止了
   */
  bool isFinished();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:

  /**
   * @brief 查看列表中是否有等待被插入的关键帧
   * @return true 如果有
   * @return false 没有
   */
  bool CheckNewKeyFrames();

  /**
   * @brief 检测回环,如果有的话就返回真
   */
  bool DetectLoop();

  /**
   * @brief 使用上一步DetectLoop()回环候选帧，计算当前关键帧的Sim3变换（sTcw），
   * 并且计算回环上所有地图点和当前帧特征点的匹配关系
   *
   * 1. 通过Bow加速描述子的匹配，利用RANSAC粗略地计算出当前帧与回环帧的特征点-地图点匹配关系和Sim3（当前帧--候选回环帧）
   * 2. 根据估计的Sim3，对3D点进行投影找到更多匹配，通过优化的方法计算更精确的Sim3（当前帧--候选回环帧）
   * 3. 找到第一个经过Sim3优化后，内点数量大于20个的候选帧，作为最好的回环帧
   * 4. 将回环帧以及回环帧相连的关键帧的地图点与当前帧的点进行匹配（当前帧--回环帧+回环帧相连关键帧）
   *    所有回环地图点写入 mvpLoopMapPoints 中，当前帧匹配上的地图点写入 mvpCurKFMatchedPoints 中
   *    （注意：实际对 mvpCurKFMatchedPoints 更新步骤见CorrectLoop()步骤3）
   * @return true         从候选帧中找到一个好回环帧，就返回true
   * @return false        当前回环不可靠
   */
  bool CheckCurKFsTcwAndLoopMPs();

  /**
   * @brief 通过将闭环时相连关键帧的MapPoints投影到这些关键帧中，进行MapPoints检查与替换
   * @param[in] CorrectedPosesMap 关联的当前帧组中的关键帧和相应的纠正后的位姿
   */
  void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

  /**
   * @brief 闭环矫正
   * 1. 通过求解的Sim3(msTcw)以及相对姿态关系，调整与当前帧相连的关键帧位姿以及这些关键帧观测到的地图点位置（相连关键帧--当前帧）
   * 2. 将闭环帧以及闭环帧相连的关键帧的地图点和与当前帧相连的关键帧的点进行匹配（当前帧+相连关键帧--闭环帧+相连关键帧）
   * 3. 通过MapPoints的匹配关系更新这些帧之间的连接关系，即更新 covisibility graph
   * 4. 对Essential Graph（Pose Graph）进行优化，MapPoints的位置则根据优化后的位姿做相对应的调整
   * 5. 创建线程进行全局Bundle Adjustment
   */
  void CorrectLoop();

  /**
   * @brief  当前线程调用,检查是否有外部线程请求复位当前线程,如果有的话就复位回环检测线程
   *
   */
  void TryResetIfRequested();

  // 是否有复位当前线程的请求
  bool mbResetRequested;

  // 是否有终止当前线程的请求
  bool mbFinishRequested;

  // 当前线程是否已经停止工作（由外部线程调用isFinished()读取）
  bool mbFinished;

  // (全局)地图的指针
  Map* mpMap;

  // 追踪线程句柄
  Tracking* mpTracker;

  // 关键帧数据库
  KeyFrameDatabase* mpKeyFrameDB;

  // 词袋模型中的大字典
  ORBVocabulary* mpORBVocabulary;

  // 局部建图线程句柄
  LocalMapping *mpLocalMapper;

  // 一个队列, 其中存储了参与到回环检测的关键帧 (当然这些关键帧也有可能因为各种原因被设置成为bad,这样虽然这个关键帧还是存储在这里但是实际上已经不再实质性地参与到回环检测的过程中去了)
  std::list<KeyFrame*> mlpLoopKeyFrameQueue;

  // 操作参与到回环检测队列中的关键帧时,使用的互斥量
  std::mutex mMutexLoopQueue;

  // Loop detector parameters
  // 连续性阈值,构造函数中将其设置成为了3
  float mnCovisibilityConsistencyTh;

  // Loop detector variables
  // 当前关键帧,其实称之为"当前正在处理的关键帧"更加合适
  KeyFrame* mpCurrentKF;

  // 最终检测出来的,和当前关键帧形成闭环的闭环关键帧
  KeyFrame* mpMatchedKF;

  // 上一次执行的时候产生的连续组
  std::vector<CandinateGroup> mvLastConsistentGroups;

  // 从上面的关键帧中进行筛选之后得到的具有足够的"连续性"的关键帧
  // 相当于更高层级的、更加优质的闭环候选帧
  std::vector<KeyFrame*> mvpBetterCandidates;

  // 下面的变量中存储的（回环关键帧的）地图点在"当前关键帧"中找到了匹配点的地图点集合
  std::vector<MapPoint*> mvpCurKFMatchedPoints;

  // 当前关键帧对应的回环上的所有地图点（来自回环帧和回环帧相邻帧的地图点）
  std::vector<MapPoint*> mvpLoopMapPoints;

  // 下面的变量的cv::Mat格式版本
  cv::Mat msTcw;

  // 当得到了当前关键帧的闭环关键帧以后,计算出来的从世界坐标系到当前帧的sim3变换
  g2o::Sim3 mg2osTcw;

  /// 上一次闭环帧的id
  long unsigned int mLastCorrectLoopKFid;

  // Variables related to Global Bundle Adjustment
  // 全局BA线程是否在进行
  bool mbRunningGBA;

  // 由当前线程调用,请求停止当前正在进行的全局BA
  // （全局BA内部会检测这一项，但是他不一定真的能停下来）
  bool mbStopGBA;

  // 全局BA线程句柄
  std::thread mThreadGBA;

  // 已经进行了的全局BA次数(包含中途被打断的)
  size_t mnFullBAIdx;
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H