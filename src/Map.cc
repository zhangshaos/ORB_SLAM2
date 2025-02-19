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


#include "MapPoint.h"
#include "KeyFrame.h"
#include "Map.h"


namespace ORB_SLAM2
{


//构造函数,地图点中最大关键帧id归0
Map::Map() : mnMaxKFid(0), mnBigChangeIdx(0)
{
}

/*
 * @brief Insert KeyFrame in the map
 * @param pKF KeyFrame
 */
//在地图中插入关键帧,同时更新关键帧的最大id
void Map::AddKeyFrame(KeyFrame *pKF)
{
  unique_lock<mutex> lock(mMutexMap);
  mspKeyFrames.insert(pKF);
  if(pKF->mnId>mnMaxKFid)
    mnMaxKFid=pKF->mnId;
}

/*
 * @brief Insert MapPoint in the map
 * @param pMP MapPoint
 */
//向地图中插入地图点
void Map::AddMapPoint(MapPoint *pMP)
{
  unique_lock<mutex> lock(mMutexMap);
  mspMapPoints.insert(pMP);
}

/**
 * @brief 从地图中删除地图点,但是其实这个地图点所占用的内存空间并没有被释放
 *
 * @param[in] pMP
 */
void Map::EraseMapPoint(MapPoint *pMP)
{
  unique_lock<mutex> lock(mMutexMap);
  mspMapPoints.erase(pMP);

  // 下面是作者加入的注释. 实际上只是从std::set中删除了地图点的指针, 原先地图点
  // 占用的内存区域并没有得到释放
  // TODO: 解决MapPoint的内存泄漏问题（只new不delete）
  // Delete the MapPoint
}

/**
 * @brief Erase KeyFrame from the map
 * @param pKF KeyFrame
 */
void Map::EraseKeyFrame(KeyFrame *pKF)
{
  unique_lock<mutex> lock(mMutexMap);
  // 是的,根据值来删除地图点
  mspKeyFrames.erase(pKF);

  // TODO: 解决KeyFrame的内存泄漏问题（只new不delete）
  // Delete the MapPoint
}

/*
 * @brief 设置参考地图点用于绘图显示局部地图点（红色），将用于DrawMapPoints函数画图
 * @param vpMPs Local MapPoints
 *
 */
void Map::SetLocalMapPoints(const vector<MapPoint *> &vpMPs)
{
  unique_lock<mutex> lock(mMutexMap);
  mvpLocalMapPoints = vpMPs;
}

//REVIEW 这个好像没有用到
void Map::InformNewBigChange()
{
  unique_lock<mutex> lock(mMutexMap);
  mnBigChangeIdx++;
}

//这个在原版的泡泡机器人注释的版本中是没有这个函数和上面的函数的
//REVIEW 目测也是当前在程序中没有被被用到过
int Map::GetLastBigChangeIdx()
{
  unique_lock<mutex> lock(mMutexMap);
  return mnBigChangeIdx;
}

//获取地图中的所有关键帧
vector<KeyFrame*> Map::GetAllKeyFrames()
{
  unique_lock<mutex> lock(mMutexMap);
  return {mspKeyFrames.begin(),mspKeyFrames.end()};
}

//获取地图中的所有地图点
vector<MapPoint*> Map::GetAllMapPoints()
{
  unique_lock<mutex> lock(mMutexMap);
  return {mspMapPoints.begin(),mspMapPoints.end()};
}

//获取地图点数目
long unsigned int Map::MapPointsInMap()
{
  unique_lock<mutex> lock(mMutexMap);
  return mspMapPoints.size();
}

//获取地图中的关键帧数目
long unsigned int Map::KeyFramesInMap()
{
  unique_lock<mutex> lock(mMutexMap);
  return mspKeyFrames.size();
}

//获取参考地图点
vector<MapPoint*> Map::GetLocalMapPoints()
{
  unique_lock<mutex> lock(mMutexMap);
  return mvpLocalMapPoints;
}

//获取地图中最大的关键帧id
long unsigned int Map::GetMaxKFid()
{
  unique_lock<mutex> lock(mMutexMap);
  return mnMaxKFid;
}

//清空地图中的数据
void Map::clear()
{
  for(auto mp : mspMapPoints)
    delete mp;

  for(auto kf : mspKeyFrames)
    delete kf;

  mspMapPoints.clear();
  mspKeyFrames.clear();
  mnMaxKFid = 0;
  mvpLocalMapPoints.clear();
  mvpKeyFrameOrigins.clear();
}


std::vector<KeyFrame *> Map::GetLastKeyFrames(unsigned int curFrame, int lastPeriod)
{
  unique_lock<mutex> lock(mMutexMap);
  std::vector<KeyFrame *> KFs;
  for (KeyFrame* KF : mspKeyFrames)
    if (KF && (curFrame - KF->mnFrameId) <= lastPeriod)
      KFs.emplace_back(KF);
  return KFs;
}


} //namespace ORB_SLAM