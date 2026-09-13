cmake_minimum_required(VERSION 3.8)
project(loco_mani_rl_controller)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/d1_piper_l_mujoco.py
  scripts/d1_piper_hardware.py
  scripts/command_file.py
  scripts/loco_mani_config.py
  scripts/loco_mani_keyboard.py
  scripts/loco_mani_command_bridge.py
  scripts/loco_mani_runtime.py
  scripts/d1_piper_l_ros2_node.py
  scripts/d1_tita_adapter.py
  scripts/loco_mani_hardware_inspect.py
  scripts/loco_mani_preflight.py
  ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_loco_mani_assets.py
  ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/check_policy_abi.py
  ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/check_d1_vendor_abi.py
  ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/compare_policy_traces.py
  ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/test_loco_mani_contract.py
  ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/export_isaaclab_trace.py
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
