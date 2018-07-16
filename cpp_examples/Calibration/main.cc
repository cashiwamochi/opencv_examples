#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "Calibration.hpp"

#include <dirent.h>

std::vector<std::string> readFileInDir(char* path_to_image) {
    std::vector<std::string> paths;
    DIR* dp=opendir(path_to_image);
    std::string s_path_to_image = path_to_image;

    if (dp!=NULL)
    {
        struct dirent* d;
        do{
            d = readdir(dp);
            if (d!=NULL) {
                std::string file_name = d->d_name;
                if(file_name == "." or file_name == ".." or file_name == ".DS_Store") continue;
                paths.push_back(s_path_to_image + file_name);
            }
        }while(d!=NULL);
    }
    closedir(dp);

    return paths;
}

int main(int argc, char* argv[])
{
  if(argc != 6) {
    std::cout <<
      "usage : this.out [/path/to/images(jpg,png)] [circle-grid or checker-board] [Row number] [Col number] [fisheye or pinhole]"
              << std::endl;
    return -1;
  }

  std::vector<std::string> vs_image_names = readFileInDir(argv[1]);
  std::vector<cv::Mat> vm_images((int)vs_image_names.size());

  for(int i = 0; i < vs_image_names.size(); i++) {
    cv::Mat image = cv::imread(vs_image_names[i], 1);
    vm_images[i] = image.clone();
  }

  std::string _pattern_type = argv[2];
  Calib::PatternType pattern_type;
  if(_pattern_type == "circle-grid") {
    pattern_type = Calib::CIRCLE_GRID;
  }
  else if(_pattern_type == "checker-board") {
    pattern_type = Calib::CHECKER_BOARD;
  }
  else {
    std::cout << "INVALID PATTERN TYPE" << std::endl;
    return -1;
  }

  std::string _row = argv[3];
  int row_num = std::stoi(_row);
  std::string _col = argv[4];
  int col_num = std::stoi(_col);

  cv::Size s{col_num, row_num};

  std::string _camera_model = argv[5];
  Calib::CameraModel camera_model;
  if(_camera_model == "pinhole") {
    camera_model = Calib::PINHOLE;
  }
  else if(_camera_model == "fisheey") {
    camera_model = Calib::FISHEYE;
  }
  else {
    std::cout << "INVALID CAMERA MODEL" << std::endl;
    return -1;
  }

  Calib::System _calib{pattern_type, s, camera_model};
  _calib.SetImages(vm_images);
  cv::Mat K, d_params;
  _calib.Run(K, d_params);
  std::vector<cv::Mat> vm_undistorted_images = _calib.GetUndistortedImages();

  for(int i = 0; i < vm_undistorted_images.size(); i++) {
    cv::imshow("CHECK", vm_undistorted_images[i]);
    int k = cv::waitKey(0);
    if(k == 'q') {
      cv::destroyWindow("CHECK");
      break;
    }
  }

  return 1;
}
