cmake_minimum_required(VERSION 3.16)
project(jbl_safety)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Threads REQUIRED)

# ROS-free logic, linked into the node and the unit tests.
add_library(safety_gate_core STATIC src/safety_gate.cpp)
target_include_directories(safety_gate_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
set_target_properties(safety_gate_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(safety_gate_node src/safety_gate_node.cpp)
target_link_libraries(safety_gate_node safety_gate_core Threads::Threads)
ament_target_dependencies(safety_gate_node geometry_msgs rclcpp sensor_msgs std_msgs)

install(TARGETS safety_gate_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)

  ament_add_gtest(test_safety_gate test/test_safety_gate.cpp)
  target_link_libraries(test_safety_gate safety_gate_core)

  add_launch_test(test/test_safety_gate_launch.py TIMEOUT 180)
endif()

ament_package()
