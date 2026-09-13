cmake_minimum_required(VERSION 3.8)
project(loco_mani_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY meshes mujoco urdf xacro
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/view_mujoco_model.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
