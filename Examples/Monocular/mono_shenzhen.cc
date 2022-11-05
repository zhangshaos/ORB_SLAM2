/**
 * Test ShenZhen in UrbanScene dataset.
 */

#include <iostream>
#include <algorithm>
#include <filesystem>
#include <optional>

#include <opencv2/imgcodecs.hpp> // cv::imread()
#include <fmt/format.h>          // fmt::format()
#include <happly/happly.h>       // write ply file
#include <glog/logging.h>        // logging
#include <toml++/toml.h>         // parse .toml
#include <Eigen/Dense>           // eigen::Matrix

#include "Tracking.h"
#include "System.h"


namespace fs = std::filesystem;


std::vector<std::string> loadImages(const std::string &file)
{
  std::vector<std::string> imagesPath;
  std::ifstream f(file);
  while (f)
  {
    std::string location;
    if (f >> location)
      imagesPath.emplace_back(std::move(location));
  }
  return imagesPath;
}

/**
 * 解析采集数据，设置相机位姿Tco（世界坐标系和相机坐标系同为前z，右x，下y），并返回。
 *
 * @param[in]  filepath 真实世界坐标系下的相机位姿。
 * @param[out] revertTransform 将原点坐标系o下点转换到真实世界坐标系。
 * @return Tco array
 */
std::vector<Sophus::SE3d> loadCamerasPose(const std::string &filepath, Eigen::Matrix3d revertTransform)
{
  std::ifstream f(filepath);
  if (!f.is_open())
  {
    LOG(INFO) << fmt::format("loadCamerasPose can not open file: {}.", filepath);
    return {};
  }
  f.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // 忽略第一行注释
  // UE4坐标系：前x，右y，上z
  // AirSim坐标系：前x，右y，下z
  // ORB_SLAM2中的坐标系：前z，右x，下y
  Eigen::Matrix3d TWaxisToCaxis; // 前x，右y，下z坐标系变成前z，右x，下y坐标系
  TWaxisToCaxis <<
                0, 1, 0,
    0, 0, 1,
    1, 0, 0;
  // 将世界坐标转换为以第一个相机朝向构建的ORB_SLAM2坐标，
  // 原点为原世界坐标原点经过上述转换后的点。
  Eigen::Matrix3d Row;
  bool isOriginInitialized = false;
  std::vector<Sophus::SE3d> Tco_s;
  while (f)
  {
    // [position] x y z [rotation] x y z w
    double x, y, z, rx, ry, rz, rw;
    f >> x >> y >> z >> rx >> ry >> rz >> rw;
    if (f)
    {
      z = -z; // UE4 => AirSim 世界坐标系
      const Eigen::Quaterniond Qwc(rw, rx, ry, rz); // UE4 的Roll、Pitch、Yaw反而是右手系，不需要变换
      const Eigen::Vector3d twc(x, y, z);
      if (!isOriginInitialized)
      {
        Row = Qwc.inverse().matrix();
        isOriginInitialized = true;
      }
      const Eigen::Matrix3d Roc = Row * Qwc; // AirSim => ORB_SLAM2
      const Eigen::Vector3d toc = TWaxisToCaxis * Row * twc;
      //LOG(INFO) << "Camera Orientation:\n" << Qwc.matrix();
      //LOG(INFO) << "Camera Position:\n" << twc;
      const Sophus::SE3d Toc(Roc, toc);
      Tco_s.emplace_back(Toc.inverse());
    }
  }
  revertTransform = Row.inverse() * TWaxisToCaxis.inverse();
  return Tco_s;
}

// Main
int Main(int argc, char* argv[])
{
  toml::table args;
  try
  {
    args = toml::parse_file(argv[1]);
  } catch (const toml::parse_error& e)
  {
    LOG(ERROR) << fmt::format("toml::parse_file cause error in {}->{}(...).", __FILE__, __FUNCTION__ );
    throw;
  }

  // Retrieve paths to images
  auto imagesFilename = loadImages(args["ImagesCollectionPath"].ref<std::string>());
  Eigen::Matrix3d revertM;
  auto camerasPosition = loadCamerasPose(args["CameraPoseCollectionPath"].ref<std::string>(),
                                         revertM);
  if (imagesFilename.size() != camerasPosition.size())
    throw std::runtime_error("Sizes of 'ImagesCollectionPath' and 'CameraPoseCollectionPath' are not equal!");

  // Create slamSystem system.
  // It initializes all system threads and gets ready to process frames.
  ORB_SLAM2::System slamSystem(args["FBoWVocabularyPath"].ref<std::string>(),
                               args["ORBSLAMConfigPath"].ref<std::string>(),
                               true);

  // Main loop
  double timeStamp = 0.0;
  for(size_t i=0, iend=imagesFilename.size(); i < iend; ++i)
  {
    this_thread::sleep_for(0.5s);

    // Read image from file
    cv::Mat im = cv::imread(imagesFilename[i], cv::IMREAD_UNCHANGED);
    if(im.empty())
    {
      LOG(INFO) << fmt::format("\nFailed to load image at: {}\n", imagesFilename[i]);
      return 1;
    }

    // Pass the image to the slamSystem system
    int trackState = slamSystem.TrackMonocularWithPose(im, (timeStamp+=0.1), camerasPosition[i]);

    // Compute the depth of each tracked key points.
    if (trackState == ORB_SLAM2::Tracking::State::OK)
    {
      const std::string trackedFilePrefix = "../../Examples/Monocular/Out";
      const std::string trackedFilename = fmt::format("{}/trackPoints{}.ply", trackedFilePrefix, i);
      slamSystem.SaveTrackedMap(trackedFilename);
    }
  }

  LOG(INFO) << "\nPress ENTER to shut down the SLAM system.\n";
  std::cin.get();

  // Stop all threads
  slamSystem.Shutdown();

  slamSystem.SaveMap("../../Examples/Monocular/Out/Map.ply");

  return 0;
}


int main(int argc, char* argv[])
{
  google::InitGoogleLogging(argv[0]);
  if (argc != 2)
  {
    LOG(INFO) << fmt::format("Usage: {} launch_toml_path.", argv[0]);
    return EXIT_FAILURE;
  }
  int r = EXIT_FAILURE;
  try
  {
    r=::Main(argc, argv);
  } catch (const std::exception &e)
  {
    LOG(ERROR) << fmt::format("Catch exception: {}.", e.what());
  } catch (...)
  {
    LOG(ERROR) << "Unknown exception.";
  }
  return r;
}