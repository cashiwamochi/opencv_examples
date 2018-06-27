#include <iostream>
#include <opencv2/core.hpp>

#define OPENCV_VERSION(a,b,c) (((a) << 16) + ((b) << 8) + (c))

int main()
{
  std::cout << "Your OpenCV Version : " << CV_VERSION << std::endl;
}
