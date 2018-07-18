
add_executable (
  cam_pose_estimation
  main.cc
)

target_link_libraries (
  cam_pose_estimation
  ${OpenCV_LIBS}
)
