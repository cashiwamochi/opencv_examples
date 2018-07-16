#include "Calibration.hpp"

namespace Calib {
  System::System(PatternType _pattern, cv::Size _size, CameraModel _model)
  : m_size(_size), me_pattern_type(_pattern), me_camera_model(_model)
  {
    mv_object_points.clear();
    for(int j=0; j<m_size.height; j++) {
      for(int i=0; i<m_size.width; i++) {
        mv_object_points.push_back(cv::Point3f((float)i, (float)j, 0.f));
      }
    }
    mb_show_process = true;
    mb_done = false;
  }

  void System::SetImages(const std::vector<cv::Mat> _vm_images) {
    mvm_original_images = _vm_images;
  }

  // main
  bool System::Run(cv::Mat& K, cv::Mat& d_params) {
    int N = (int)mvm_original_images.size();
    mvm_processed_images.reserve(N);

    int detected_count = 0;

    for(int i = 0; i < N; i++) {
      cv::Mat m_processed_image = mvm_original_images[i].clone();
      std::vector<cv::Point2f> v_detected_points;
      bool found = false;
      if(me_pattern_type==CHECKER_BOARD) {
        found = cv::findChessboardCorners(m_processed_image, m_size, v_detected_points);
      }
      else if(me_pattern_type==CIRCLE_GRID) {
        found = cv::findCirclesGrid(m_processed_image, m_size, v_detected_points);
      }

      if(found) {
        drawChessboardCorners(m_processed_image, m_size, cv::Mat(v_detected_points), found);
        mvv_all_detected_points.push_back(v_detected_points);
        mvv_all_object_points.push_back(mv_object_points);
        detected_count++;
      }

      mvm_processed_images.push_back(m_processed_image);

      ShowProcess(mb_show_process, i);
    }

    if(mvv_all_detected_points.size() < 10) {
      std::cout << "There are not enough data. It needs good images (easy to detect patterns) more than 10." << std::endl;
      mb_done = false;
    }
    else {
      std::vector<cv::Mat> rvecs, tvecs;
      cv::calibrateCamera(mvv_all_object_points, mvv_all_detected_points, m_size, mmK, mm_dist, rvecs, tvecs);
      K = mmK.clone();
      d_params = mm_dist.clone();
      std::cout << "K : \n" << K << std::endl;
      std::cout << "dist-params : " << d_params << std::endl;
      mb_done = true;
    }

    cv::destroyAllWindows();
    return mb_done;
  }

  std::vector<cv::Mat> System::GetProcessedImages() {
    if(mvm_processed_images.size() > 0) {
      return mvm_processed_images;
    }
    else {
      std::vector<cv::Mat> vm_dummy(1);
      vm_dummy[0] = cv::Mat::zeros(256,256,CV_8U);
      std::cout << "The system has not proccesed " << std::endl;
      return vm_dummy;
    }
  }

  std::vector<cv::Mat> System::GetUndistortedImages() {
    if(mb_done) {
      std::vector<cv::Mat> vm_undistorted_images((int)mvm_original_images.size());
      for(int i = 0; i < (int)mvm_original_images.size(); i++) {
        cv::Mat m_undistorted_image;
        cv::undistort(mvm_original_images[i], m_undistorted_image, mmK, mm_dist);
        vm_undistorted_images[i] = m_undistorted_image.clone();
      }
      return vm_undistorted_images;
    }
    else {
      std::vector<cv::Mat> vm_dummy(1);
      vm_dummy[0] = cv::Mat::zeros(256,256,CV_8U);
      std::cout << "The system has not proccesed " << std::endl;
      return vm_dummy;
    }
  }

  void System::ShowProcess(bool& flag, const int idx) {
    if(flag) {
      cv::imshow("Processd Image(Press q to exit debug mode)", mvm_processed_images[idx].clone());
      int k = cv::waitKey(0);
      if(k == 'q') {
        flag = false;
        cv::destroyWindow("Processd Image(Press q to exit debug mode)");
      }
    }
    else {
      return;
    }
  }
}
