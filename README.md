# opencv_examples

## Examples (Implemented)

### CameraCalibration
It supports pinhole and fisheye camera models and undistortion is alse performed. Of course, both of chess-board-detection and circle-pattern-detection are available.   ```datasets``` folder includes 2 image-sequences. (Fisheye-calibration is implemented, but I didn't check this is correct.)   
A checker-board is from https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf .

### Stiching
This shows how to use cv::Sticher. This is almost same as here ( https://docs.opencv.org/master/d8/d19/tutorial_stitcher.html )

### Triangulation
This code generates 3d-points and camera poses by 2-viewes-geometry. You can learn how to use FindEssentialMat(), recoverPose(), triangulatePoints() and so on. In addtion, I made a viewer using OpenCV-Viz-Module. You can check it too.

### Version
This shows your opencv version.

### Panorama (Python)
It uses findHomography(), and seamlessCone() for blening. You can try with images in ```test_images``` folder.

### CameraCapture (Python)
Nothing to write.

## Examples (Not Yet Implemeted)

### OpticalFlow

### Aruco

### Camera Pose Estimation


### GoodFeaturesToTrack
