
add_executable (
  triangulation
  main.cc
)

target_link_libraries (
  triangulation
  ${OpenCV_LIBS}
)
