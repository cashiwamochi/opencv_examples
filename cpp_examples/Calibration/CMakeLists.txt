add_executable (
  show_version
  main.cc
)

target_link_libraries (
  show_version
  ${OpenCV_LIBS}
)
