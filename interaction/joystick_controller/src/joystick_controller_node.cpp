#include "joystick_controller/joystick_controller_node.hpp"

#include <unordered_map>
#include <iostream>
#include <fstream>
#include <cstring>
#include <linux/joystick.h>
#include <fcntl.h>
#include <unistd.h>

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

JoyStickControllerNode::JoyStickControllerNode(const rclcpp::NodeOptions & options)
: Node("joystick_controller_node", options)
{
  LOG(INFO) << "JoyStickControllNode Construct";
}

bool JoyStickControllerNode::init(const std::string& config_file) 
{
  try  {
    YAML::Node config = YAML::LoadFile(config_file);
    // Uart Interface Name
    device_ = config["joystick_controller"]["uart_interface"].as<std::string>();
    // Cmd Vel Topic Name
    cmd_vel_topic_ = config["joystick_controller"]["cmd_vel_topic"].as<std::string>();
    // Cmd Key Topic Name
    cmd_key_topic_ = config["joystick_controller"]["cmd_key_topic"].as<std::string>();
    // Params
    params_.max_vx = config["joystick_controller"]["max_vx"].as<double>();
    params_.max_vy = config["joystick_controller"]["max_vy"].as<double>();
    params_.max_w = config["joystick_controller"]["max_w"].as<double>();
  } catch(YAML::Exception& e) {
    LOG(ERROR) << "config parse failed: " << e.what();
    return false;
  }

  // init qos
  rclcpp::QoS qos(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
  qos.history(RMW_QOS_POLICY_HISTORY_KEEP_LAST).keep_last(10);

  // publishers
  cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
    cmd_vel_topic_, qos);
  fsm_goal_publisher_ =
    this->create_publisher<std_msgs::msg::String>(cmd_key_topic_, qos);
  realtime_cmd_vel_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<geometry_msgs::msg::Twist>>(
      cmd_vel_publisher_);
  realtime_fsm_goal_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::String>>(fsm_goal_publisher_);

  // init cmd vel
  for (size_t i = 0; i < 3; i++) {
    normalized_cmd_[i] = 0.0;
  }
  fsm_goal_.data = "idle";

  // init fd
  if(!fd_init_(device_)) {
    LOG(ERROR) << "fd init failed";
    return false;
  }
  return true;
}

bool JoyStickControllerNode::start() 
{
  LOG(INFO) << "JoyStickController start";
  joytick_listen_flag_ = true;
  joystick_listening_thread_ = std::thread(
    std::bind(&JoyStickControllerNode::joystick_process_,
    this)
  );
  
  publish_flag_ = true;
  publish_thread_ = std::thread(
    std::bind(&JoyStickControllerNode::publish_process_,
    this)
  );

  return true;
}

bool JoyStickControllerNode::stop() 
{
  LOG(INFO) << "JoyStickController stop";
  joytick_listen_flag_ = false;
  if(joystick_listening_thread_.joinable()) {
    joystick_listening_thread_.join();
  }

  publish_flag_ = false;
  if(publish_thread_.joinable()) {
    publish_thread_.join();
  }
  return true;
}

bool JoyStickControllerNode::fd_init_(const std::string& dev_name)
{
  // Get fd
  joy_fd_ = open(device_.c_str(), O_RDONLY);
  if (joy_fd_ < 0) {
      LOG(ERROR) << "Can't open joystick : " << device_;
      return false;
  }

  // Get dev name
  char name[128];
  if (ioctl(joy_fd_, JSIOCGNAME(sizeof(name)), name) < 0) {
      strcpy(name, "Unknown Joystick");
  }
  LOG(INFO) << "Device Name= " << name;

  // Get Axes and Button num
  int axes = 0, buttons = 0;
  ioctl(joy_fd_, JSIOCGAXES, &axes);
  ioctl(joy_fd_, JSIOCGBUTTONS, &buttons);
  LOG(INFO) << "Axes num = " << axes << ", Button num = " << buttons;
  LOG(INFO) << "Joystick fd init success";
  return true;
}

void JoyStickControllerNode::joystick_process_() 
{
  // LOG(INFO) << "joystick process";
  struct js_event event;
  while (joytick_listen_flag_) {
    ssize_t bytes = read(joy_fd_, &event, sizeof(event));
    if (bytes == sizeof(event)) {
        switch (event.type & ~JS_EVENT_INIT) {
            case JS_EVENT_BUTTON:
                button_handle_((int)event.number, event.value);
                break;
            case JS_EVENT_AXIS:
                axis_handle_((int)event.number, event.value);
                break;
        }
    }
  }
}

void JoyStickControllerNode::publish_process_()
{
  // LOG(INFO) << "publish process";

  while (publish_flag_) {
    if (realtime_cmd_vel_publisher_ && realtime_cmd_vel_publisher_->trylock()) {
      realtime_cmd_vel_publisher_->msg_ = twist_;
      realtime_cmd_vel_publisher_->unlockAndPublish();
    }
    if (realtime_fsm_goal_publisher_ && realtime_fsm_goal_publisher_->trylock()) {
      realtime_fsm_goal_publisher_->msg_ = fsm_goal_;
      realtime_fsm_goal_publisher_->unlockAndPublish();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

void JoyStickControllerNode::axis_handle_(int number, signed short value) 
{
  // Only Handle x y yaw, using left hand axis
  double diff = 0.0;
  // Handle y
  if (number == 0) {
    diff = value - JOY_AXIS_H_ZERO_VALUE;
    normalized_cmd_[1] = -1.0 *params_.max_vy * (diff  / JOY_AXIS_MAX_VALUE);
  }
  // Handle x
  if (number == 1) {
    diff = value - JOY_AXIS_V_ZERO_VALUE;
    normalized_cmd_[0] = -1.0 * params_.max_vx * (diff / JOY_AXIS_MAX_VALUE);
  }
  // Handle yaw
  if (number == 2) {
    diff = value - JOY_AXIS_H_ZERO_VALUE;
    normalized_cmd_[2] = -1.0 * params_.max_w * (diff / JOY_AXIS_MAX_VALUE);
  }
  // LOG(INFO) << "normalized cmd = " << normalized_cmd_[0] << " " << normalized_cmd_[1] << " " << normalized_cmd_[2];

  twist_.linear.x = normalized_cmd_[0];
  twist_.linear.y = normalized_cmd_[1];
  twist_.linear.z = 0.0;
  twist_.angular.z = normalized_cmd_[2];
}

void JoyStickControllerNode::button_handle_(int number, signed short value)
{
  // LOG(INFO) << "button handle: " << "number = " << number << " value = " << value;
  std::string event_name = button_mapping_[number];
  // LOG(INFO) << "button event = " << event_name;
  fsm_goal_.data = event_name;
}