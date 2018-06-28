#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
  if(argc == 5) {
    std::cout <<
      "usage : this.out [/path/to/images(jpg,png)] [circle-board or checker-board] [Row number] [Col number] [fisheye or pinhole]"
              << std::endl;
    return -1;
  }


  return 1;
}
