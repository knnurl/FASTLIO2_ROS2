#pragma once

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <nav_msgs/msg/path.hpp>

#include "fastlio2_validation/srv/load_ground_truth.hpp"
#include "fastlio2_validation/srv/compare_to_baseline.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <vector>
#include <mutex>
#include <string>

namespace fastlio2_validation
{

struct Pose3D
{
  double timestamp;
  Eigen::Vector3d translation;
  Eigen::Quaterniond rotation;
};

class MapValidator : public rclcpp::Node
{
public:
  explicit MapValidator(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  // Callbacks
  void onEstimatedPath(const nav_msgs::msg::Path::SharedPtr msg);
  void onGroundTruthPath(const nav_msgs::msg::Path::SharedPtr msg);
  void timerCallback();

  // Services
  void computeNow(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> res);
  void setBaseline(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> res);
  void compareToBaseline(
    const std::shared_ptr<fastlio2_validation::srv::CompareToBaseline::Request> req,
    std::shared_ptr<fastlio2_validation::srv::CompareToBaseline::Response> res);
  void loadGroundTruth(
    const std::shared_ptr<fastlio2_validation::srv::LoadGroundTruth::Request> req,
    std::shared_ptr<fastlio2_validation::srv::LoadGroundTruth::Response> res);

  // Metrics computation
  void computeMetrics();

  // ATE: Horn's method for SE3 alignment
  static Eigen::Matrix4d hornAlignment(const std::vector<Eigen::Vector3d>& src,
                                        const std::vector<Eigen::Vector3d>& dst);
  double computeATE(const std::vector<Pose3D>& estimated,
                    const std::vector<Pose3D>& ground_truth);
  double computeATERMSE(const std::vector<Pose3D>& estimated,
                         const std::vector<Pose3D>& ground_truth);

  // RPE
  struct RPEResult { double mean_trans; double mean_rot_deg; };
  RPEResult computeRPE(const std::vector<Pose3D>& trajectory, double delta_m);

  // Load trajectory from file
  std::vector<Pose3D> loadTrajectoryTUM(const std::string& path);

  // Parameters
  double delta_m_;
  double ate_regression_threshold_;

  // State
  std::mutex mutex_;
  std::vector<Pose3D> estimated_trajectory_;
  std::vector<Pose3D> ground_truth_trajectory_;
  bool has_ground_truth_ = false;

  // Current metrics
  double current_ate_rmse_ = 0.0;
  double current_rpe_trans_ = 0.0;
  double current_rpe_rot_ = 0.0;

  // Baseline metrics (for pre/post merge comparison)
  double baseline_ate_rmse_ = 0.0;
  double baseline_rpe_trans_ = 0.0;
  double baseline_rpe_rot_ = 0.0;
  bool has_baseline_ = false;

  // ROS interfaces
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_estimated_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_ground_truth_;

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_ate_rmse_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_rpe_trans_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_rpe_rot_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_report_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_compute_now_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_set_baseline_;
  rclcpp::Service<fastlio2_validation::srv::CompareToBaseline>::SharedPtr srv_compare_baseline_;
  rclcpp::Service<fastlio2_validation::srv::LoadGroundTruth>::SharedPtr srv_load_gt_;

  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace fastlio2_validation
