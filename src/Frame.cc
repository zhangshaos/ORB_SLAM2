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


#include <thread>

#include "MapPoint.h"
#include "ORBVocabulary.h"
#include "ORBmatcher.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "Frame.h"


namespace ORB_SLAM2
{


//下一个生成的帧的ID,这里是初始化类的静态成员变量
long unsigned int Frame::nNextId = 0;

//是否要进行初始化操作的标志
//这里给这个标志置位的操作是在最初系统开始加载到内存的时候进行的，下一帧就是整个系统的第一帧，所以这个标志要置位
bool Frame::mbInitialComputations = true;


float Frame::cx = 0.f,
  Frame::cy = 0.f,
  Frame::fx = 0.f,
  Frame::fy = 0.f,
  Frame::invfx = 0.f,
  Frame::invfy = 0.f;
float Frame::mnMinX = 0.f,
  Frame::mnMinY = 0.f,
  Frame::mnMaxX = 0.f,
  Frame::mnMaxY = 0.f;
float Frame::mfGridElementWidthInv = 0.f,
  Frame::mfGridElementHeightInv = 0.f;


Frame::Frame() {}


Frame::Frame(const Frame &frame)
  :mpORBvocabulary(frame.mpORBvocabulary),
   mpORBextractorLeft(frame.mpORBextractorLeft),
   mpORBextractorRight(frame.mpORBextractorRight),
   mTimeStamp(frame.mTimeStamp),
   mK(frame.mK.clone()),									      //深拷贝
   mDistCoef(frame.mDistCoef.clone()),					//深拷贝
   N(frame.N),
   mvKeys(frame.mvKeys),									      //经过实验，确定这种通过同类型对象初始化的操作是具有深拷贝的效果的
   mvKeysRight(frame.mvKeysRight), 						  //深拷贝
   mvKeysUn(frame.mvKeysUn),  							    //深拷贝
   mBowVec(frame.mBowVec), 								      //深拷贝
   mFeatVec(frame.mFeatVec),								    //深拷贝
   mDescriptors(frame.mDescriptors.clone()), 				    //cv::Mat深拷贝
   mDescriptorsRight(frame.mDescriptorsRight.clone()),	//cv::Mat深拷贝
   mvpMapPoints(frame.mvpMapPoints), 						//深拷贝
   mnId(frame.mnId),
   mpReferenceKF(frame.mpReferenceKF),
   mnScaleLevels(frame.mnScaleLevels),
   mfScaleFactor(frame.mfScaleFactor),
   mfLogScaleFactor(frame.mfLogScaleFactor),
   mvScaleFactors(frame.mvScaleFactors), 					//深拷贝
   mvInvScaleFactors(frame.mvInvScaleFactors),    //深拷贝
   mvLevelSigma2(frame.mvLevelSigma2), 					  //深拷贝
   mvInvLevelSigma2(frame.mvInvLevelSigma2)				//深拷贝
{
  //逐个复制，其实这里也是深拷贝
  for(int i=0;i<FRAME_GRID_COLS;i++)
    for(int j=0; j<FRAME_GRID_ROWS; j++)
      //这里没有使用前面的深拷贝方式的原因可能是mGrid是由若干vector类型对象组成的vector，
      //但是自己不知道vector内部的源码不清楚其赋值方式，在第一维度上直接使用上面的方法可能会导致
      //错误使用不合适的复制函数，导致第一维度的vector不能够被正确地“拷贝”
      mGrid[i][j]=frame.mGrid[i][j];

  if(!frame.mTcw.empty())
    //这里说的是给新的帧设置Pose
    SetPose(frame.mTcw);
}


/**
 * @brief 单目帧构造函数
 *
 * @param[in] imGray                            //灰度图
 * @param[in] timeStamp                         //时间戳
 * @param[in & out] extractor                   //ORB特征点提取器的句柄
 * @param[in] voc                               //ORB字典的句柄
 * @param[in] K                                 //相机的内参数矩阵
 * @param[in] distCoef                          //相机的去畸变参数
 */
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef)
  :mpORBvocabulary(voc),
   mpORBextractorLeft(extractor),
   mpORBextractorRight(nullptr),
   mTimeStamp(timeStamp),
   mK(K.clone()),
   mDistCoef(distCoef.clone())
{
  // Frame ID
  // Step 1 帧的ID 自增
  mnId=nNextId++;

  // Step 2 计算图像金字塔的参数
  // Scale Level Info
  //获取图像金字塔的层数
  mnScaleLevels = mpORBextractorLeft->GetLevels();
  //获取每层的缩放因子
  mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
  //计算每层缩放因子的自然对数
  mfLogScaleFactor = log(mfScaleFactor);
  //获取各层图像的缩放因子
  mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
  //获取各层图像的缩放因子的倒数
  mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
  //获取sigma^2
  mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
  //获取sigma^2的倒数
  mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

  // ORB extraction
  // Step 3 对这个单目图像进行提取特征点, 第一个参数0-左图， 1-右图
  ExtractORB(imGray);

  //求出特征点的个数
  N = mvKeys.size();

  //如果没有能够成功提取出特征点，那么就直接返回了
  if(mvKeys.empty())
    return;

  // Step 4 用OpenCV的矫正函数、内参对提取到的特征点进行矫正
  UndistortKeyPoints();

  // 初始化本帧的地图点
  mvpMapPoints = vector<MapPoint*>(N, nullptr);

  // This is done only for the first Frame (or after a change in the calibration)
  //  Step 5 计算去畸变后图像边界，将特征点分配到网格中。这个过程一般是在第一帧或者是相机标定参数发生变化之后进行
  if(mbInitialComputations)
  {
    // 计算去畸变后图像的边界
    ComputeImageBounds(imGray);

    // 表示一个图像像素相当于多少个图像网格列（宽）
    mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
    // 表示一个图像像素相当于多少个图像网格行（高）
    mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

    //给类的静态成员变量复制
    fx = K.at<float>(0,0);
    fy = K.at<float>(1,1);
    cx = K.at<float>(0,2);
    cy = K.at<float>(1,2);
    // 猜测是因为这种除法计算需要的时间略长，所以这里直接存储了这个中间计算结果
    invfx = 1.0f/fx;
    invfy = 1.0f/fy;

    //特殊的初始化过程完成，标志复位
    mbInitialComputations=false;
  }

  // 将特征点分配到图像网格中
  AssignFeaturesToGrid();
}

/**
 * @brief 将提取的ORB特征点分配到图像网格中
 *
 */
void Frame::AssignFeaturesToGrid()
{
  // Step 1  给存储特征点的网格数组 Frame::mGrid 预分配空间
  // FRAME_GRID_COLS = 64，FRAME_GRID_ROWS=48
  int nReserve = 0.5f * N / (FRAME_GRID_COLS*FRAME_GRID_ROWS);
  //开始对mGrid这个二维数组中的每一个vector元素遍历并预分配空间
  for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
    for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
      mGrid[i][j].reserve(nReserve);

  // Step 2 遍历每个特征点，将每个特征点在mvKeysUn中的索引值放到对应的网格mGrid中
  for(int i=0; i<N; i++)
  {
    //从类的成员变量中获取已经去畸变后的特征点
    const cv::KeyPoint &kp = mvKeysUn[i];

    //存储某个特征点所在网格的网格坐标，nGridPosX范围：[0,FRAME_GRID_COLS], nGridPosY范围：[0,FRAME_GRID_ROWS]
    int nGridPosX, nGridPosY;
    // 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX,nGridPosY里，返回true，没找到返回false
    if(PosInGrid(kp, nGridPosX, nGridPosY))
      //如果找到特征点所在网格坐标，将这个特征点的索引添加到对应网格的数组mGrid中
      mGrid[nGridPosX][nGridPosY].push_back(i);
  }
}

/**
 * @brief 提取图像的ORB特征点，提取的关键点存放在mvKeys，描述子存放在mDescriptors
 *
 * @param[in] im            等待提取特征点的图像
 */
void Frame::ExtractORB(const cv::Mat &im)
{
  // 左图的话就套使用左图指定的特征点提取器，并将提取结果保存到对应的变量中
  // 这里使用了仿函数来完成，重载了括号运算符 ORBextractor::operator()
  (*mpORBextractorLeft)(im,				      //待提取特征点的图像
                        cv::Mat(),		  //掩摸图像, 实际没有用到
                        mvKeys,			    //输出变量，用于保存提取后的特征点
                        mDescriptors);	//输出变量，用于保存特征点的描述子
}

// 设置相机姿态
void Frame::SetPose(cv::Mat Tcw)
{
  mTcw = Tcw.clone();
  UpdatePoseMatrices();
}

// 根据Tcw计算mRcw、mtcw和mRwc、mOw
void Frame::UpdatePoseMatrices()
{
  // mOw：    当前相机光心在世界坐标系下坐标
  // mTcw：   世界坐标系到相机坐标系的变换矩阵
  // mRcw：   世界坐标系到相机坐标系的旋转矩阵
  // mtcw：   世界坐标系到相机坐标系的平移向量
  // mRwc：   相机坐标系到世界坐标系的旋转矩阵

  //从变换矩阵中提取出旋转矩阵
  //注意，rowRange这个只取到范围的左边界，而不取右边界
  mRcw = mTcw.rowRange(0,3).colRange(0,3);

  // mRcw求逆即可
  mRwc = mRcw.t();

  // 从变换矩阵中提取出旋转矩阵
  mtcw = mTcw.rowRange(0,3).col(3);

  // mTcw 求逆后是当前相机坐标系变换到世界坐标系下，对应的光心变换到世界坐标系下就是 mTcw的逆 中对应的平移向量
  mOw = -mRcw.t()*mtcw;
}

/**
 * @brief 判断地图点是否在视野中
 * 步骤
 * Step 1 获得这个地图点的世界坐标，经过以下层层关卡的判断，通过的地图点才认为是在视野中
 * Step 2 关卡一：将这个地图点变换到当前帧的相机坐标系下，如果深度值为正才能继续下一步。
 * Step 3 关卡二：将地图点投影到当前帧的像素坐标，如果在图像有效范围内才能继续下一步。
 * Step 4 关卡三：计算地图点到相机中心的距离，如果在有效距离范围内才能继续下一步。
 * Step 5 关卡四：计算当前相机指向地图点向量和地图点的平均观测方向夹角，小于60°才能进入下一步。
 * Step 6 根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
 * Step 7 记录计算得到的一些参数
 * @param[in] pMP                       当前地图点
 * @param[in] viewingCosLimit           当前相机指向地图点向量和地图点的平均观测方向夹角余弦阈值
 * @return true                         地图点合格，且在视野内
 * @return false                        地图点不合格，抛弃
 */
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
  // mbNeedTrackInView 是决定一个地图点是否进行重投影的标志
  // 这个标志的确定要经过多个函数的确定，isInFrustum()只是其中的一个验证关卡。这里默认设置为否
  pMP->mbNeedTrackInView = false;

  // 3D in absolute coordinates
  // Step 1 获得这个地图点的世界坐标
  cv::Mat P = pMP->GetWorldPos();

  // 3D in camera coordinates
  // 根据当前帧(粗糙)位姿转化到当前相机坐标系下的三维点Pc
  const cv::Mat Pc = mRcw*P + mtcw;
  const float &PcX = Pc.at<float>(0);
  const float &PcY = Pc.at<float>(1);
  const float &PcZ = Pc.at<float>(2);

  // Check positive depth
  // Step 2 关卡一：将这个地图点变换到当前帧的相机坐标系下，如果深度值为正才能继续下一步。
  if(PcZ < 0.0f)
    return false;

  // Project in image and check it is not outside
  // Step 3 关卡二：将地图点投影到当前帧的像素坐标，如果在图像有效范围内才能继续下一步。
  const float invz = 1.0f / PcZ;
  const float u = fx*PcX*invz + cx;
  const float v = fy*PcY*invz + cy;

  // 判断是否在图像边界中，只要不在那么就说明无法在当前帧下进行重投影
  if(u<mnMinX || u>mnMaxX)
    return false;
  if(v<mnMinY || v>mnMaxY)
    return false;

  // Check distance is in the scale invariance region of the MapPoint
  // Step 4 关卡三：计算地图点到相机中心的距离，如果在有效距离范围内才能继续下一步。
  // 得到认为的可靠距离范围:[0.8f*mfMinDistance, 1.2f*mfMaxDistance]
  const float maxDistance = pMP->GetMaxDistanceInvariance();
  const float minDistance = pMP->GetMinDistanceInvariance();

  // 得到当前地图点距离当前帧相机光心的距离,注意P，mOw都是在同一坐标系下才可以
  //  mOw：当前相机光心在世界坐标系下坐标
  const cv::Mat PO = P - mOw;
  // 取模就得到了距离
  const float dist = cv::norm(PO);

  // 如果不在有效范围内，认为投影不可靠
  if(dist < minDistance || dist > maxDistance)
    return false;

  // Check viewing angle
  // Step 5 关卡四：计算当前相机指向地图点向量和地图点的平均观测方向夹角，小于60°才能进入下一步。
  cv::Mat Pn = pMP->GetNormal();

  // 计算当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值，注意平均观测方向为单位向量
  const float viewCos = PO.dot(Pn) / dist;

  //夹角要在60°范围内，否则认为观测方向太偏了，重投影不可靠，返回false
  if(viewCos<viewingCosLimit)
    return false;

  // Predict scale in the image
  // Step 6 根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
  const int nPredictedLevel = pMP->PredictScale(dist,		//这个点到光心的距离
                                                this);
  // Step 7 记录计算得到的一些参数
  // Data used by the tracking
  // 通过置位标记 MapPoint::mbNeedTrackInView 来表示这个地图点要被投影
  pMP->mbNeedTrackInView = true;

  // 该地图点投影在当前图像（一般是左图）的像素横坐标
  pMP->mTrackedProjX = u;

  // 该地图点投影在当前图像（一般是左图）的像素纵坐标
  pMP->mTrackedProjY = v;

  // 根据地图点到光心距离，预测的该地图点的尺度层级
  pMP->mnTrackedScaleLevel = nPredictedLevel;

  // 保存当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值
  pMP->mTrackedViewCosine = viewCos;

  //执行到这里说明这个地图点在相机的视野中并且进行重投影是可靠的，返回true
  return true;
}

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
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
  // 存储搜索结果的vector
  vector<size_t> vIndices;
  vIndices.reserve(N);

  // Step 1 计算半径为r圆左右上下边界所在的网格列和行的id
  // 查找半径为r的圆左侧边界所在网格列坐标。这个地方有点绕，慢慢理解下：
  // (mnMaxX-mnMinX)/FRAME_GRID_COLS：表示列方向每个网格可以平均分得几个像素（肯定大于1）
  // mfGridElementWidthInv=FRAME_GRID_COLS/(mnMaxX-mnMinX) 是上面倒数，表示每个像素可以均分几个网格列（肯定小于1）
  // (x-mnMinX-r)，可以看做是从图像的左边界mnMinX到半径r的圆的左边界区域占的像素列数
  // 两者相乘，就是求出那个半径为r的圆的左侧边界在哪个网格列中
  // 保证nMinCellX 结果大于等于0
  const int nMinCellX = max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));

  // 如果最终求得的圆的左边界所在的网格列超过了设定了上限，那么就说明计算出错，找不到符合要求的特征点，返回空vector
  if(nMinCellX>=FRAME_GRID_COLS)
    return vIndices;

  // 计算圆所在的右边界网格列索引
  const int nMaxCellX = min((int)FRAME_GRID_COLS-1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
  // 如果计算出的圆右边界所在的网格不合法，说明该特征点不好，直接返回空vector
  if(nMaxCellX<0)
    return vIndices;

  //后面的操作也都是类似的，计算出这个圆上下边界所在的网格行的id
  const int nMinCellY = max(0, (int)floor((y-mnMinY-r) * mfGridElementHeightInv));
  if(nMinCellY>=FRAME_GRID_ROWS)
    return vIndices;

  const int nMaxCellY = min((int)FRAME_GRID_ROWS-1, (int)ceil((y-mnMinY+r) * mfGridElementHeightInv));
  if(nMaxCellY<0)
    return vIndices;

  // 检查需要搜索的图像金字塔层数范围是否符合要求
  //? 疑似bug。(minLevel>0) 后面条件 (maxLevel>=0)肯定成立
  //? 改为 const bool bCheckLevels = (minLevel>=0) || (maxLevel>=0);
  const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

  // Step 2 遍历圆形区域内的所有网格，寻找满足条件的候选特征点，并将其index放到输出里
  for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
  {
    for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
    {
      // 获取这个网格内的所有特征点在 Frame::mvKeysUn 中的索引
      const vector<size_t> vCell = mGrid[ix][iy];
      // 如果这个网格中没有特征点，那么跳过这个网格继续下一个
      if(vCell.empty())
        continue;

      // 如果这个网格中有特征点，那么遍历这个图像网格中所有的特征点
      for(unsigned long long j : vCell)
      {
        // 根据索引先读取这个特征点
        const cv::KeyPoint &kpUn = mvKeysUn[j];
        // 保证给定的搜索金字塔层级范围合法
        if(bCheckLevels)
        {
          // cv::KeyPoint::octave中表示的是从金字塔的哪一层提取的数据
          // 保证特征点是在金字塔层级minLevel和maxLevel之间，不是的话跳过
          if(kpUn.octave<minLevel)
            continue;
          if(maxLevel>=0)		//? 为何特意又强调？感觉多此一举
            if(kpUn.octave>maxLevel)
              continue;
        }

        // 通过检查，计算候选特征点到圆中心的距离，查看是否是在这个圆形区域之内
        const float distx = kpUn.pt.x-x;
        const float disty = kpUn.pt.y-y;

        // 如果x方向和y方向的距离都在指定的半径之内，存储其index为候选特征点
        if(fabs(distx)<r && fabs(disty)<r)
          vIndices.push_back(j);
      }
    }
  }
  return vIndices;
}


/**
 * @brief 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX,nGridPosY里，返回true，没找到返回false
 *
 * @param[in] kp                    给定的特征点
 * @param[in & out] posX            特征点所在网格坐标的横坐标
 * @param[in & out] posY            特征点所在网格坐标的纵坐标
 * @return true                     如果找到特征点所在的网格坐标，返回true
 * @return false                    没找到返回false
 */
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
  // 计算特征点x,y坐标落在哪个网格内，网格坐标为posX，posY
  // mfGridElementWidthInv=(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
  // mfGridElementHeightInv=(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);
  posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
  posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

  //Keypoint's coordinates are undistorted, which could cause to go out of the image
  // 因为特征点进行了去畸变，而且前面计算是round取整，所以有可能得到的点落在图像网格坐标外面
  // 如果网格坐标posX，posY超出了[0,FRAME_GRID_COLS] 和[0,FRAME_GRID_ROWS]，表示该特征点没有对应网格坐标，返回false
  if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
    return false;

  // 计算成功返回true
  return true;
}

/**
 * @brief 计算当前帧特征点对应的词袋Bow，主要是mBowVec 和 mFeatVec
 *
 */
void Frame::ComputeBoW()
{
  // 判断是否以前已经计算过了，计算过了就跳过
  if(mBowVec.empty())
  {
    // 将描述子mDescriptors转换为DBOW要求的输入格式
    vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
    // 将特征点的描述子转换成词袋向量mBowVec以及特征向量mFeatVec
    mpORBvocabulary->transform(vCurrentDesc,	//当前的描述子vector
                               mBowVec,			  //输出，词袋向量，记录的是单词的id及其对应权重TF-IDF值
                               mFeatVec,		  //输出，记录node id及其对应的图像 feature对应的索引
                               4);				    //4表示从叶节点向前数的层数
  }
}

/**
 * @brief 用内参对特征点去畸变，结果报存在mvKeysUn中
 *
 */
void Frame::UndistortKeyPoints()
{
  // Step 1 如果第一个畸变参数为0，不需要矫正。第一个畸变参数k1是最重要的，一般不为0，为0的话，说明畸变参数都是0
  // 变量mDistCoef中存储了opencv指定格式的去畸变参数，格式为：(k1,k2,p1,p2,k3)
  if(mDistCoef.at<float>(0)==0.0)
  {
    mvKeysUn=mvKeys;
    return;
  }


  // Step 2 如果畸变参数不为0，用OpenCV函数进行畸变矫正
  // Fill matrix with points
  // N为提取的特征点数量，为满足OpenCV函数输入要求，将N个特征点保存在N*2的矩阵中
  cv::Mat mat(N,2,CV_32F);
  //遍历每个特征点，并将它们的坐标保存到矩阵中
  for(int i=0; i<N; i++)
  {
    //然后将这个特征点的横纵坐标分别保存
    mat.at<float>(i,0)=mvKeys[i].pt.x;
    mat.at<float>(i,1)=mvKeys[i].pt.y;
  }

  // Undistort points
  // 函数reshape(int cn,int rows=0) 其中cn为更改后的通道数，rows=0表示这个行将保持原来的参数不变
  //为了能够直接调用opencv的函数来去畸变，需要先将矩阵调整为2通道（对应坐标x,y）
  mat=mat.reshape(2);
  cv::undistortPoints(mat,				//输入的特征点坐标
                      mat,				//输出的校正后的特征点坐标覆盖原矩阵
                      mK,					//相机的内参数矩阵
                      mDistCoef,	//相机畸变参数矩阵
                      cv::Mat(),	//一个空矩阵，对应为函数原型中的R
                      mK); 				//新内参数矩阵，对应为函数原型中的P

  //调整回只有一个通道，回归我们正常的处理方式
  mat=mat.reshape(1);

  // Fill undistorted keypoint vector
  // Step 存储校正后的特征点
  mvKeysUn.resize(N);
  //遍历每一个特征点
  for(int i=0; i<N; i++)
  {
    //根据索引获取这个特征点
    //注意之所以这样做而不是直接重新声明一个特征点对象的目的是，能够得到源特征点对象的其他属性
    cv::KeyPoint kp = mvKeys[i];
    //读取校正后的坐标并覆盖老坐标
    kp.pt.x=mat.at<float>(i,0);
    kp.pt.y=mat.at<float>(i,1);
    mvKeysUn[i]=kp;
  }
}

/**
 * @brief 计算去畸变图像的边界
 *
 * @param[in] imLeft            需要计算边界的图像
 */
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
  // 如果畸变参数不为0，用OpenCV函数进行畸变矫正
  if(mDistCoef.at<float>(0)!=0.0)
  {
    // 保存矫正前的图像四个边界点坐标： (0,0) (cols,0) (0,rows) (cols,rows)
    cv::Mat mat(4,2,CV_32F);
    mat.at<float>(0,0)=0.0;         //左上
    mat.at<float>(0,1)=0.0;
    mat.at<float>(1,0)=imLeft.cols; //右上
    mat.at<float>(1,1)=0.0;
    mat.at<float>(2,0)=0.0;         //左下
    mat.at<float>(2,1)=imLeft.rows;
    mat.at<float>(3,0)=imLeft.cols; //右下
    mat.at<float>(3,1)=imLeft.rows;

    // Undistort corners
    // 和前面校正特征点一样的操作，将这几个边界点作为输入进行校正
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    //校正后的四个边界点已经不能够围成一个严格的矩形，因此在这个四边形的外侧加边框作为坐标的边界
    mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));//左上和左下横坐标最小的
    mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));//右上和右下横坐标最大的
    mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));//左上和右上纵坐标最小的
    mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));//左下和右下纵坐标最小的
  }
  else
  {
    // 如果畸变参数为0，就直接获得图像边界
    mnMinX = 0.0f;
    mnMaxX = imLeft.cols;
    mnMinY = 0.0f;
    mnMaxY = imLeft.rows;
  }
}

} //namespace ORB_SLAM