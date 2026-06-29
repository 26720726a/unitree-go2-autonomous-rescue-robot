#include <chrono>
#include <memory>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "unitree_api/msg/request.hpp"

#include "common/ros2_sport_client.h"

using namespace std::chrono_literals;

class CmdVelClassicWalk : public rclcpp::Node
{
public:
  CmdVelClassicWalk() : Node("cmd_vel_classic_walk")
  {
    req_pub_ = this->create_publisher<unitree_api::msg::Request>(
      "/api/sport/request", 10);

    cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10,
      std::bind(&CmdVelClassicWalk::cmdVelCallback, this, std::placeholders::_1));

    init_timer_ = this->create_wall_timer(
      500ms,
      std::bind(&CmdVelClassicWalk::initSequence, this));

    watchdog_timer_ = this->create_wall_timer(
      100ms,
      std::bind(&CmdVelClassicWalk::watchdog, this));

    RCLCPP_INFO(this->get_logger(), "cmd_vel_classic_walk node started");
  }

private:
  void publishRequest(unitree_api::msg::Request &req)
  {
    req_pub_->publish(req);
  }

  void initSequence()
  {
    unitree_api::msg::Request req;

    if (init_step_ == 0)
    {
      RCLCPP_INFO(this->get_logger(), "Step 1: StandUp");
      sport_client_.StandUp(req);
      publishRequest(req);
      init_step_++;
      return;
    }

    if (init_step_ < 7)
    {
      init_step_++;
      return;
    }

    if (init_step_ == 7)
    {
      RCLCPP_INFO(this->get_logger(), "Step 2: ClassicWalk");
      sport_client_.ClassicWalk(req);
      publishRequest(req);
      init_step_++;
      return;
    }

    if (init_step_ < 10)
    {
      init_step_++;
      return;
    }

    RCLCPP_INFO(this->get_logger(), "ClassicWalk ready. Nav2 cmd_vel is now accepted.");
    nav2_enabled_ = true;
    init_timer_->cancel();
  }

  void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    last_cmd_time_ = this->now();

    latest_vx_ = clamp(msg->linear.x,  -0.25, 0.25);
    latest_vy_ = clamp(msg->linear.y,  -0.15, 0.15);
    latest_wz_ = clamp(msg->angular.z, -0.50, 0.50);

    has_cmd_ = true;

    if (!nav2_enabled_)
    {
      return;
    }

    publishMove(latest_vx_, latest_vy_, latest_wz_);
  }

  void publishMove(double vx, double vy, double wz)
  {
    unitree_api::msg::Request req;
    sport_client_.Move(req, vx, vy, wz);
    publishRequest(req);
  }

  void publishStop()
  {
    unitree_api::msg::Request req;
    sport_client_.StopMove(req);
    publishRequest(req);
  }

  void watchdog()
  {
    if (!nav2_enabled_)
    {
      return;
    }

    if (!has_cmd_)
    {
      return;
    }

    double dt = (this->now() - last_cmd_time_).seconds();

    if (dt > 0.5)
    {
      publishStop();
      has_cmd_ = false;
    }
  }

  double clamp(double value, double min_value, double max_value)
  {
    return std::max(min_value, std::min(value, max_value));
  }

private:
  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr req_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::TimerBase::SharedPtr init_timer_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;

  SportClient sport_client_;

  int init_step_ = 0;
  bool nav2_enabled_ = false;
  bool has_cmd_ = false;

  double latest_vx_ = 0.0;
  double latest_vy_ = 0.0;
  double latest_wz_ = 0.0;

  rclcpp::Time last_cmd_time_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CmdVelClassicWalk>());
  rclcpp::shutdown();
  return 0;
}
