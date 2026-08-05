cmake_minimum_required(VERSION 3.8)
project(autoware_shoulder_pullover_manager)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-O3 -Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(autoware_adapi_v1_msgs REQUIRED)
find_package(autoware_common_msgs REQUIRED)
find_package(autoware_perception_msgs REQUIRED)
find_package(autoware_planning_msgs REQUIRED)
find_package(tier4_system_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_library(shoulder_pullover_manager_lib
  src/goal_scorer.cpp
  src/dubins_path.cpp
  src/curvature_spiral_path.cpp
  src/trajectory_planner.cpp
  src/keyboard_trigger.cpp
  src/pull_over_manager_node.cpp
)
target_include_directories(shoulder_pullover_manager_lib PUBLIC
  include
)
ament_target_dependencies(shoulder_pullover_manager_lib
  rclcpp
  nav_msgs
  geometry_msgs
  builtin_interfaces
  autoware_adapi_v1_msgs
  autoware_common_msgs
  autoware_perception_msgs
  autoware_planning_msgs
  tier4_system_msgs
  visualization_msgs
)

add_executable(shoulder_pullover_manager_node
  src/main.cpp
)
target_link_libraries(shoulder_pullover_manager_node shoulder_pullover_manager_lib)

install(TARGETS
  shoulder_pullover_manager_node
  DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS
  shoulder_pullover_manager_lib
  DESTINATION lib
)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
