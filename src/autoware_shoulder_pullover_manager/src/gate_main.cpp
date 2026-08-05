#include "autoware_shoulder_pullover_manager/trajectory_gate_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto node = std::make_shared<autoware::shoulder_pullover_manager::TrajectoryGateNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
