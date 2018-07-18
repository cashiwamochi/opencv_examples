#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
  if(argc<2) {
    std::cout << "[usage] this.out [image1] [image2] ..." << std::endl;
    return -1;
  }

  std::vector<cv::Mat> vm_images;
  for(int i = 1; i < argc; i++) {
    std::string image_name = argv[i];
    cv::Mat image = cv::imread(image_name, 1);
    vm_images.push_back(image);
  }

  cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA;

  cv::Mat panorama;
  cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(mode, false);
  cv::Stitcher::Status status = stitcher->stitch(vm_images, panorama);

  if (status != cv::Stitcher::OK)
  {
    std::cout << "Can't stitch images, error code = " << int(status) << std::endl;
    return -1;
  }

  cv::imshow("panorama(press q to exit)", panorama);

  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}
