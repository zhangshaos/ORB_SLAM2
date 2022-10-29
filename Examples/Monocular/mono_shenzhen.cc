/**
 * Test ShenZhen in UrbanScene dataset.
 */

#include <iostream>
#include <algorithm>
#include <filesystem>
#include <opencv2/core/core.hpp> // cv::imread()
#include <fmt/format.h>          // fmt::format()
#include <happly/happly.h>       // write ply file
#include <glog/logging.h>        // logging
#include <toml++/toml.h>         // parse .toml
#include <Eigen/Dense>           // eigen::Matrix

#include "System.h"

using namespace std;
namespace fs = std::filesystem;


vector<string> loadImages(const string &file)
{
  vector<string> imagesPath;
  ifstream f(file);
  while (f)
  {
    std::string location;
    if (f >> location)
      imagesPath.emplace_back(std::move(location));
  }
  return imagesPath;
}

vector<Eigen::Vector3d> loadCamerasPose(const string &filepath, bool src_is_ue4= true)
{
  ifstream f(filepath);
  if (!f.is_open())
  {
    LOG(INFO) << fmt::format("loadCamerasPose can not open file: {}.", filepath);
    return {};
  }
  f.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // 忽略第一行注释
  vector<Eigen::Vector3d> positions;
  while (f)
  {
    // [position] x y z [rotation] x y z w
    double x, y, z, rx, ry, rz, rw;
    f >> x >> y >> z >> rx >> ry >> rz >> rw;
    if (src_is_ue4)
    {
      // ORB-SLAM是右手系，而UE4是左手系，AirSim是右手系
      double t = y;
      y = z;
      z = t;
    }
    if (f)
      positions.emplace_back(x, y, z);
  }
  return positions;
}

// Main
int Main(int argc, char* argv[])
{
  toml::table args;
  try {
    args = toml::parse_file(argv[1]);
  } catch (const toml::parse_error& e) {
    LOG(ERROR) << fmt::format("toml::parse_file cause error in {}->{}(...).", __FILE__, __FUNCTION__ );
    throw;
  }

  // Retrieve paths to images
  vector<string> imagesFilename = ::loadImages(args["ImagesCollectionPath"].ref<std::string>());
  vector<Eigen::Vector3d> camerasPosition = ::loadCamerasPose(args["CameraPoseCollectionPath"].ref<std::string>());
  assert(imagesFilename.size() == camerasPosition.size());

  // Create slamSystem system. It initializes all system threads and gets ready to process frames.
  ORB_SLAM2::System slamSystem(args["FBoWVocabularyPath"].ref<std::string>(),
                               args["ORBSLAMConfigPath"].ref<std::string>(),
                               ORB_SLAM2::System::MONOCULAR, true);


  // Main loop
  double timeStamp = 0.0;
  for(int i=0, iend=imagesFilename.size(); i < iend; ++i)
  {
    // Pause for mapping thread to generate more map points.
    std::this_thread::sleep_for(0.5s);

    // Read image from file
    cv::Mat im = cv::imread(imagesFilename[i], cv::IMREAD_UNCHANGED);
    if(im.empty())
    {
      LOG(INFO) << fmt::format("\nFailed to load image at: {}\n", imagesFilename[i]);
      return 1;
    }

    // Pass the image to the slamSystem system
    Eigen::Matrix4d poseTcw;
    cv::Mat Tcw = slamSystem.TrackMonocular(im, (timeStamp += 0.1));
    if (!Tcw.empty())
    {
      poseTcw << Tcw.at<float>(0, 0), Tcw.at<float>(0, 1), Tcw.at<float>(0, 2), Tcw.at<float>(0, 3),
                 Tcw.at<float>(1, 0), Tcw.at<float>(1, 1), Tcw.at<float>(1, 2), Tcw.at<float>(1, 3),
                 Tcw.at<float>(2, 0), Tcw.at<float>(2, 1), Tcw.at<float>(2, 2), Tcw.at<float>(2, 3),
                 Tcw.at<float>(3, 0), Tcw.at<float>(3, 1), Tcw.at<float>(3, 2), Tcw.at<float>(3, 3);
      LOG(INFO) << "poseTcw is:\n" << poseTcw << endl;
    }

    // Compute the depth of each tracked key points.
    if (slamSystem.GetTrackingState() == ORB_SLAM2::Tracking::eTrackingState::OK)
    {
      const std::string trackedFilePrefix = "../../Examples/Monocular/Out";
      const std::string trackedFilename = fmt::format("{}/trackPoints{}.ply", trackedFilePrefix, i);
      std::ofstream trackedFile(trackedFilename);
      if (!trackedFile.is_open())
      {
        LOG(INFO) << fmt::format("Write to {} failed. It is not open.\n", trackedFilename);
        continue;
      }
      auto keyPoints = slamSystem.GetTrackedKeyPointsUn();
      auto mapPoints = slamSystem.GetTrackedMapPoints();
      assert(keyPoints.size() == mapPoints.size());
      // Save the results as .ply file.
      happly::PLYData plyFile;
      std::vector<std::array<double, 3>> vertexesPos;
      std::vector<std::array<uchar, 3>> vertexesColor;
      std::vector<float> vertexesPtX;
      std::vector<float> vertexesPtY;
      std::vector<uchar> vertexesPtOctave;
      const int N = keyPoints.size();
      for (int i=0; i<N; ++i)
      {
        auto* mpt = mapPoints[i];
        const auto& kpt = keyPoints[i];
        if (mpt && !mpt->isBad())
        {
          cv::Mat wpos = mpt->GetWorldPos();
          Eigen::Matrix<double, 4, 1> wpos_homo; wpos_homo << wpos.at<float>(0), wpos.at<float>(1), wpos.at<float>(2), 1.f;
          Eigen::Matrix<double, 4, 1> camPos = poseTcw * wpos_homo;
          // LOG(INFO) << '\n' << poseTcw << "\n@\n" << wpos_homo << "\n=\n" << camPos;
          vertexesPos.emplace_back(std::array<double,3>{ camPos.x(), camPos.y(), camPos.z() });
          vertexesPtX.emplace_back(kpt.pt.x);
          vertexesPtY.emplace_back(kpt.pt.y);
          vertexesPtOctave.emplace_back(kpt.octave);
          const auto& ptColor = im.at<cv::Vec3b>(kpt.pt.y, kpt.pt.x);
          vertexesColor.emplace_back(std::array<uchar,3>{ptColor[2], ptColor[1], ptColor[0]});  // r, g, b
        }
      }
      const size_t M = vertexesPos.size();
      plyFile.addVertexPositions(vertexesPos);
      plyFile.addVertexColors(vertexesColor);
      plyFile.getElement("vertex").addProperty<float>("ix", vertexesPtX);
      plyFile.getElement("vertex").addProperty<float>("iy", vertexesPtY);
      plyFile.getElement("vertex").addProperty<uchar>("octave", vertexesPtOctave);
      plyFile.write(trackedFile);
    }
  }

  LOG(INFO) << "\nPress ENTER to shut down the SLAM system.\n";
  std::cin.get();

  // Stop all threads
  slamSystem.Shutdown();

  // Save camera trajectory
  slamSystem.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

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
  int r=EXIT_FAILURE;
  try {
    r=::Main(argc, argv);
  } catch (const std::exception &e) {
    LOG(ERROR) << fmt::format("Catch exception: {}.", e.what());
  } catch (...) {
    LOG(ERROR) << "Unknown exception.";
  }
  return r;
}