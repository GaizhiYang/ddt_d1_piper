#include <cmath>
#include <string>
#include <memory>
#include <unordered_map>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/parameter_client.hpp"
#include "rcl_interfaces/srv/set_parameters.hpp"
#include "realtime_tools/realtime_publisher.hpp"

#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/twist.hpp"


using namespace std::chrono_literals;

#define JOY_AXIS_MAX_VALUE 32767
#define JOY_AXIS_MIN_VALUE -32767
#define JOY_AXIS_H_ZERO_VALUE 565
#define JOY_AXIS_V_ZERO_VALUE -476

typedef struct 
{
  double max_vx = 0.0;
  double max_vy = 0.0;
  double max_w = 0.0;
} JoyStickParams;

typedef std::unordered_map<int, std::string> ButtonMap;

class JoyStickControllerNode : public rclcpp::Node
{
public:
  JoyStickControllerNode(const rclcpp::NodeOptions & options);
  bool init(const std::string& config_file);
  bool start();
  bool stop();

private:
  bool fd_init_(const std::string& dev_nam);
  bool is_connect_();

  void axis_handle_(int number, signed short value);
  void button_handle_(int number, signed short value);

  void joystick_process_();
  void publish_process_();

  double normalized_cmd_[3];

  std::thread joystick_listening_thread_;
  std::thread publish_thread_;

  std::string device_ = "";
  int joy_fd_ = -1;
  bool joytick_listen_flag_ = false;
  bool publish_flag_ = false;

  // Publisher
  std::string cmd_vel_topic_ = "";
  std::string cmd_key_topic_ = "";

  std::shared_ptr<rclcpp::Publisher<geometry_msgs::msg::Twist>> cmd_vel_publisher_;
  std::shared_ptr<rclcpp::Publisher<std_msgs::msg::String>> fsm_goal_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::Twist>>
    realtime_cmd_vel_publisher_;
  std::shared_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::String>>
    realtime_fsm_goal_publisher_;

  // Msgs
  geometry_msgs::msg::Twist twist_;
  std_msgs::msg::String fsm_goal_;

  // JoyStick Params
  JoyStickParams params_;

  // Button Mapping
  ButtonMap button_mapping_ = {
    {0, "rl_0"}, {1, "rl_1"}, {2, "rl_2"}, {3, "rl_3"},
    {5, "transform_up"}, {6, "transform_down"}, {4, "idle"}
  };
};