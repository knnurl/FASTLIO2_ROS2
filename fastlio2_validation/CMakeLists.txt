cmake_minimum_required(VERSION 3.8)
project(fastlio2_validation)

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
find_package(pcl_conversions REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(Eigen3 REQUIRED)

# ── Service generation ────────────────────────────────────────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/LoadGroundTruth.srv"
  "srv/CompareToBaseline.srv"
  DEPENDENCIES std_msgs
)

# ── Include directories ──────────────────────────────────────────────────────
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# ── Map validator node ────────────────────────────────────────────────────────
add_executable(map_validator
  src/map_validator.cpp
)
ament_target_dependencies(map_validator
  rclcpp std_msgs std_srvs nav_msgs sensor_msgs geometry_msgs pcl_conversions
)
target_link_libraries(map_validator ${PCL_LIBRARIES})
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)
target_link_libraries(map_validator "${cpp_typesupport_target}")

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS map_validator
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
