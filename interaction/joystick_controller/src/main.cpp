#include <atomic>
#include <thread>
#include <chrono>

#include <glog/logging.h>
#include "joystick_controller/joystick_controller_node.hpp"


int main(int argc, char ** argv)
{
  // std::signal(SIGINT, signalHandler);
  rclcpp::init(argc, argv);
  auto options = rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<JoyStickControllerNode>(options);
  
  std::string config_file = "/home/yjyflashing/ros2_workspace/ddt_controller_ws/src/ddt_ros2_control/interaction/joystick_controller/config/param.yaml";
  if(!node->init(config_file)) {
    LOG(INFO) << "JoyStick init failed";
    return -1;
  }
  if(!node->start()) {
    LOG(INFO) << "JoyStick start failed";
    return -1;
  }
  rclcpp::spin(node);
  node->stop();
  rclcpp::shutdown();
  return 0;
}