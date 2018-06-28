#include <iostream>
#include <vector>
#include <string>

// OpenCV
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char* argv[]) {
  if(argc != 3) {
    cout <<
      "usage: this.out [/path/to/image1] [/path/to/image2] "
         << endl;
    return -1;
  }

  cv::Mat image1 = cv::imread(argv[1]);
  cv::Mat image2 = cv::imread(argv[2]);

  // Camera intristic parameter matrix
  // I did not calibration
  cv::Mat K = (cv::Mat_<float>(3,3) <<  1000.f,   0.f, image1.cols / 2.f,
                                          0.f, 1000.f, image1.rows / 2.f,
                                          0.f,   0.f,               1.f);

  vector<cv::KeyPoint> kpts_vec1, kpts_vec2;
  cv::Mat desc1, desc2;
  cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();

  // extract feature points and calculate descriptors
  akaze -> detectAndCompute(image1, cv::noArray(), kpts_vec1, desc1);
  akaze -> detectAndCompute(image2, cv::noArray(), kpts_vec2, desc2);


  cv::BFMatcher* matcher = new cv::BFMatcher(cv::NORM_HAMMING, false);
  // cross check flag set to false
  // because i do cross-ratio-test match
  vector< vector<cv::DMatch> > matches_2nn_12, matches_2nn_21;
  matcher->knnMatch( desc1, desc2, matches_2nn_12, 2 );
  matcher->knnMatch( desc2, desc1, matches_2nn_21, 2 );
  const double ratio = 0.8;

  vector<cv::Point2f> selected_points1, selected_points2;

  for(int i = 0; i < matches_2nn_12.size(); i++) { // i is queryIdx
    if( matches_2nn_12[i][0].distance/matches_2nn_12[i][1].distance < ratio
        and
        matches_2nn_21[matches_2nn_12[i][0].trainIdx][0].distance
          / matches_2nn_21[matches_2nn_12[i][0].trainIdx][1].distance < ratio )
    {
      if(matches_2nn_21[matches_2nn_12[i][0].trainIdx][0].trainIdx
            == matches_2nn_12[i][0].queryIdx)
      {
        selected_points1.push_back(kpts_vec1[matches_2nn_12[i][0].queryIdx].pt);
        selected_points2.push_back(
            kpts_vec2[matches_2nn_21[matches_2nn_12[i][0].trainIdx][0].queryIdx].pt
            );
      }
    }
  }

  if(false) {
    cv::Mat src;
    cv::hconcat(image1, image2, src);
    for(int i = 0; i < selected_points1.size(); i++) {
      cv::line( src, selected_points1[i],
                cv::Point2f(selected_points2[i].x + image1.cols, selected_points2[i].y),
                1, 1, 0 );
    }
    cv::imwrite("match-result.png", src);
  }

  cv::Mat Kd;
  K.convertTo(Kd, CV_64F);

  cv::Mat mask; // unsigned char array
  cv::Mat E = cv::findEssentialMat(selected_points1, selected_points2, Kd.at<double>(0,0),
                           // cv::Point2f(0.f, 0.f),
                           cv::Point2d(image1.cols/2., image1.rows/2.),
                           cv::RANSAC, 0.999, 1.0, mask);
  // E is CV_64F not 32F

  vector<cv::Point2f> inlier_match_points1, inlier_match_points2;
  for(int i = 0; i < mask.rows; i++) {
    if(mask.at<unsigned char>(i)){
      inlier_match_points1.push_back(selected_points1[i]);
      inlier_match_points2.push_back(selected_points2[i]);
    }
  }

  if(false) {
    cv::Mat src;
    cv::hconcat(image1, image2, src);
    for(int i = 0; i < inlier_match_points1.size(); i++) {
      cv::line( src, inlier_match_points1[i],
                cv::Point2f(inlier_match_points2[i].x + image1.cols, inlier_match_points2[i].y),
                1, 1, 0 );
    }
    cv::imwrite("inlier_matches.png", src);
  }

  mask.release();
  cv::Mat R, t;
  cv::recoverPose(E,
                  inlier_match_points1,
                  inlier_match_points2,
                  R, t, Kd.at<double>(0,0),
                  // cv::Point2f(0, 0),
                  cv::Point2d(image1.cols/2.0, image1.rows/2.0),
                  mask);
  // R,t is CV_64F not 32F

  vector<cv::Point2d> triangulation_points1, triangulation_points2;
  for(int i = 0; i < mask.rows; i++) {
    if(mask.at<unsigned char>(i)){
      triangulation_points1.push_back
                   (cv::Point2d((double)inlier_match_points1[i].x,(double)inlier_match_points1[i].y));
      triangulation_points2.push_back
                   (cv::Point2d((double)inlier_match_points2[i].x,(double)inlier_match_points2[i].y));
    }
  }

  if(true) {
    cv::Mat src;
    cv::hconcat(image1, image2, src);
    for(int i = 0; i < triangulation_points1.size(); i++) {
      cv::line( src, triangulation_points1[i],
                cv::Point2f((float)triangulation_points2[i].x + (float)image1.cols,
                            (float)triangulation_points2[i].y),
                1, 1, 0 );
    }
    cv::imwrite("triangulated-points.png", src);
  }

  cv::Mat Rt0 = cv::Mat::eye(3, 4, CV_64FC1);
  cv::Mat Rt1 = cv::Mat::eye(3, 4, CV_64FC1);
  R.copyTo(Rt1.rowRange(0,3).colRange(0,3));
  t.copyTo(Rt1.rowRange(0,3).col(3));


  cv::Mat point3d_homo;
  cv::triangulatePoints(Kd * Rt0, Kd * Rt1,
                        triangulation_points1, triangulation_points2,
                        point3d_homo);
  //point3d_homo is 64F [4x3] (homogenious)
  //available input type is here
  //https://stackoverflow.com/questions/16295551/how-to-correctly-use-cvtriangulatepoints

  assert(point3d_homo.cols == triangulation_points1.size());

  std::cout << "Map Point Num : " << point3d_homo.cols << std::endl;

#if 1
  // OpenCV viz module Viewer
  {
    vector<cv::Mat> v_camposes(2);
    v_camposes[0] = Rt0; v_camposes[1] = Rt1;
    vector<cv::Vec3d> vPointPosition;
    vPointPosition.reserve(point3d_homo.cols);
    for(size_t c = 0; c < point3d_homo.cols; c++) {
      vPointPosition.push_back(cv::Vec3d(point3d_homo.at<double>(0,c)/point3d_homo.at<double>(3,c),
                                         point3d_homo.at<double>(1,c)/point3d_homo.at<double>(3,c),
                                         point3d_homo.at<double>(2,c)/point3d_homo.at<double>(3,c)));
    }

    cv::viz::Viz3d myWindow("OpenCV-Viewer");

    cv::viz::WCloud wcloud(vPointPosition, cv::viz::Color::yellow());
    wcloud.setRenderingProperty( cv::viz::POINT_SIZE, 2 );
    myWindow.showWidget("Viewer", wcloud);

    myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(0.3));
    cv::viz::WCameraPosition cpw(0.3); // Coordinate axes

    for(size_t i = 0; i < v_camposes.size(); i++) {
      cv::Affine3d cam_pose;
      cv::Mat R, t;
      R = v_camposes[i].rowRange(0,3).colRange(0,3);
      t = v_camposes[i].rowRange(0,3).col(3);
      cv::Mat T = -R.t()*t;
      cam_pose = cv::Affine3d(R.t(), T);
      cv::viz::WCameraPosition cpw_frustums(cv::Matx33d(Kd), /*image,*/ 0.4, cv::viz::Color::green()); // Camera frustum
      string widgetPoseName = "CPW" + std::to_string(i);
      string widgetFrustumName = "CPW_FRUSTUM" + std::to_string(i);
      myWindow.showWidget(widgetPoseName, cpw, cam_pose);
      myWindow.showWidget(widgetFrustumName, cpw_frustums, cam_pose);
    }
    std::cout << "Press q to quit viewer." << std::endl;
    myWindow.spin();
  }
#endif

  return 0;
}
