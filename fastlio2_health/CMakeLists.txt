cmake_minimum_required(VERSION 3.8)
project(fastlio2_health)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)
find_package(Eigen3 REQUIRED)
find_package(fastlio2_validation REQUIRED)

# ── Message generation ────────────────────────────────────────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MapHealthStatus.msg"
  "msg/HBATrigger.msg"
  "msg/HBACompletion.msg"
  "msg/MergeTrigger.msg"
  "msg/MergeCompletion.msg"
  "msg/LoopClosureEvent.msg"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs nav_msgs
)

# ── Include directories ──────────────────────────────────────────────────────
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# ── Pose graph partitioner library ───────────────────────────────────────────
add_library(pose_graph_partitioner SHARED
  src/pose_graph_partitioner.cpp
)
target_include_directories(pose_graph_partitioner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)
ament_target_dependencies(pose_graph_partitioner rclcpp)

# ── Map health monitor node ──────────────────────────────────────────────────
add_executable(map_health_monitor
  src/map_health_monitor.cpp
  src/map_staleness_tracker.cpp
)
ament_target_dependencies(map_health_monitor
  rclcpp std_msgs std_srvs nav_msgs sensor_msgs geometry_msgs
  tf2_ros tf2_geometry_msgs pcl_conversions fastlio2_validation
)
target_link_libraries(map_health_monitor
  pose_graph_partitioner
  ${PCL_LIBRARIES}
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)
target_link_libraries(map_health_monitor "${cpp_typesupport_target}")

# ── Voxel merge node ─────────────────────────────────────────────────────────
add_executable(voxel_merge_node
  src/voxel_merge_node.cpp
)
ament_target_dependencies(voxel_merge_node
  rclcpp std_msgs sensor_msgs geometry_msgs pcl_conversions
)
target_link_libraries(voxel_merge_node
  ${PCL_LIBRARIES}
  "${cpp_typesupport_target}"
)

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS map_health_monitor voxel_merge_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS pose_graph_partitioner
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# ── Testing ──────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(pose_graph_partitioner)
ament_export_dependencies(rclcpp Eigen3 PCL)
ament_package()
