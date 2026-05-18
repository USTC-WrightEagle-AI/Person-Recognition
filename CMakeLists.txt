cmake_minimum_required(VERSION 3.0.2)
project(cade_vision)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  tf2_ros
)

catkin_python_setup()
catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  scripts/open_vision_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY src/cade_vision/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
