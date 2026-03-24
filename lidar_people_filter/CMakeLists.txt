cmake_minimum_required(VERSION 3.8)
project(lidar_people_filter)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  lidar_people_filter/dbscan_filter_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME dbscan_filter_node
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
