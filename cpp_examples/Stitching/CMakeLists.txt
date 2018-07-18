
add_executable (
  stitching
  main.cc
)

target_link_libraries (
  stitching
  ${OpenCV_LIBS}
)
