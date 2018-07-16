#include <vector>
#include <opencv2/opencv.hpp>


namespace Calib{

  enum CameraModel {
    PINHOLE=0,
    FISHEYE=1
    };

  enum PatternType {
    CHECKER_BOARD=0,
    CIRCLE_GRID=1
    };

  class System {
  public:
    System(PatternType _pattern, cv::Size _size, CameraModel _model);
    ~System(){};
    void SetImages(const std::vector<cv::Mat> _vm_images);
    bool Run(cv::Mat& K, cv::Mat& d_params);
    std::vector<cv::Mat> GetProcessedImages();
    std::vector<cv::Mat> GetUndistortedImages();

  private:
    std::vector<cv::Mat> mvm_original_images;
    std::vector<cv::Mat> mvm_processed_images;
    const cv::Size m_size;
    const PatternType me_pattern_type;
    const CameraModel me_camera_model;
    std::vector<cv::Point3f> mv_object_points;
    std::vector< std::vector<cv::Point2f> > mvv_all_detected_points;
    std::vector< std::vector<cv::Point3f> > mvv_all_object_points;

    cv::Mat mmK;
    cv::Mat mm_dist;

    bool mb_done;

    bool mb_show_process;
    void ShowProcess(bool& flag, const int idx);
  };

}
