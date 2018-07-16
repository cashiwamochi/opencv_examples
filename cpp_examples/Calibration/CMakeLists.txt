add_executable (
  calibrate_camera
  main.cc
  Calibration.cc
)

target_link_libraries (
  calibrate_camera
  ${OpenCV_LIBS}
)
