#include "fastlio2_validation/map_validator.hpp"

#include <Eigen/SVD>
#include <sstream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace fastlio2_validation
{

// ─────────────────────────────────────────────────────────────────────────────
MapValidator::MapValidator(const rclcpp::NodeOptions& options)
  : Node("map_validator", options)
{
  this->declare_parameter("delta_m", 1.0);
  this->declare_parameter("ate_regression_threshold", 0.05);
  this->declare_parameter("ground_truth_pcd_file", "");

  delta_m_ = this->get_parameter("delta_m").as_double();
  ate_regression_threshold_ = this->get_parameter("ate_regression_threshold").as_double();

  auto default_qos = rclcpp::SystemDefaultsQoS();

  sub_estimated_ = this->create_subscription<nav_msgs::msg::Path>(
    "/fastlio2/lio_path", default_qos,
    std::bind(&MapValidator::onEstimatedPath, this, std::placeholders::_1));

  sub_ground_truth_ = this->create_subscription<nav_msgs::msg::Path>(
    "/ground_truth/path", default_qos,
    std::bind(&MapValidator::onGroundTruthPath, this, std::placeholders::_1));

  pub_ate_rmse_ = this->create_publisher<std_msgs::msg::Float64>(
    "/validation/ate_rmse", default_qos);
  pub_rpe_trans_ = this->create_publisher<std_msgs::msg::Float64>(
    "/validation/rpe_trans", default_qos);
  pub_rpe_rot_ = this->create_publisher<std_msgs::msg::Float64>(
    "/validation/rpe_rot", default_qos);
  pub_report_ = this->create_publisher<std_msgs::msg::String>(
    "/validation/report", default_qos);

  srv_compute_now_ = this->create_service<std_srvs::srv::Trigger>(
    "/validation/compute_now",
    std::bind(&MapValidator::computeNow, this,
              std::placeholders::_1, std::placeholders::_2));

  srv_set_baseline_ = this->create_service<std_srvs::srv::Trigger>(
    "/validation/set_baseline",
    std::bind(&MapValidator::setBaseline, this,
              std::placeholders::_1, std::placeholders::_2));

  srv_compare_baseline_ = this->create_service<fastlio2_validation::srv::CompareToBaseline>(
    "/validation/compare_to_baseline",
    std::bind(&MapValidator::compareToBaseline, this,
              std::placeholders::_1, std::placeholders::_2));

  srv_load_gt_ = this->create_service<fastlio2_validation::srv::LoadGroundTruth>(
    "/validation/load_ground_truth",
    std::bind(&MapValidator::loadGroundTruth, this,
              std::placeholders::_1, std::placeholders::_2));

  // Periodic computation at 0.2 Hz
  timer_ = this->create_wall_timer(
    std::chrono::seconds(5),
    std::bind(&MapValidator::timerCallback, this));

  RCLCPP_INFO(this->get_logger(), "MapValidator initialised [delta=%.1fm, ate_threshold=%.3fm]",
              delta_m_, ate_regression_threshold_);
}

// ─────────────────────────────────────────────────────────────────────────────
void MapValidator::onEstimatedPath(const nav_msgs::msg::Path::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  estimated_trajectory_.clear();
  estimated_trajectory_.reserve(msg->poses.size());
  for (const auto& ps : msg->poses) {
    Pose3D p;
    p.timestamp = rclcpp::Time(ps.header.stamp).seconds();
    p.translation = Eigen::Vector3d(
      ps.pose.position.x, ps.pose.position.y, ps.pose.position.z);
    p.rotation = Eigen::Quaterniond(
      ps.pose.orientation.w, ps.pose.orientation.x,
      ps.pose.orientation.y, ps.pose.orientation.z);
    estimated_trajectory_.push_back(p);
  }
}

void MapValidator::onGroundTruthPath(const nav_msgs::msg::Path::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  ground_truth_trajectory_.clear();
  ground_truth_trajectory_.reserve(msg->poses.size());
  for (const auto& ps : msg->poses) {
    Pose3D p;
    p.timestamp = rclcpp::Time(ps.header.stamp).seconds();
    p.translation = Eigen::Vector3d(
      ps.pose.position.x, ps.pose.position.y, ps.pose.position.z);
    p.rotation = Eigen::Quaterniond(
      ps.pose.orientation.w, ps.pose.orientation.x,
      ps.pose.orientation.y, ps.pose.orientation.z);
    ground_truth_trajectory_.push_back(p);
  }
  has_ground_truth_ = !ground_truth_trajectory_.empty();
}

void MapValidator::timerCallback()
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (estimated_trajectory_.size() < 2) return;
  computeMetrics();
}

// ─────────────────────────────────────────────────────────���───────────────────
// Services
// ─────────────────────────────────────────────────────────────────────────────

void MapValidator::computeNow(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (estimated_trajectory_.size() < 2) {
    res->success = false;
    res->message = "Not enough trajectory data";
    return;
  }
  computeMetrics();
  res->success = true;
  res->message = "Metrics computed";
}

void MapValidator::setBaseline(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
  std::lock_guard<std::mutex> lock(mutex_);
  baseline_ate_rmse_ = current_ate_rmse_;
  baseline_rpe_trans_ = current_rpe_trans_;
  baseline_rpe_rot_ = current_rpe_rot_;
  has_baseline_ = true;
  res->success = true;
  res->message = "Baseline set: ATE=" + std::to_string(baseline_ate_rmse_) +
                 " RPE_t=" + std::to_string(baseline_rpe_trans_);
  RCLCPP_INFO(this->get_logger(), "Baseline set: ATE_RMSE=%.4f, RPE_t=%.4f, RPE_r=%.2f",
              baseline_ate_rmse_, baseline_rpe_trans_, baseline_rpe_rot_);
}

void MapValidator::compareToBaseline(
  const std::shared_ptr<fastlio2_validation::srv::CompareToBaseline::Request>,
  std::shared_ptr<fastlio2_validation::srv::CompareToBaseline::Response> res)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!has_baseline_) {
    res->passed = false;
    res->ate_delta = 0.0;
    res->rpe_trans_delta = 0.0;
    res->rpe_rot_delta = 0.0;
    res->report_json = "{\"error\":\"No baseline set\"}";
    return;
  }

  computeMetrics();

  double ate_delta = current_ate_rmse_ - baseline_ate_rmse_;
  double rpe_trans_delta = current_rpe_trans_ - baseline_rpe_trans_;
  double rpe_rot_delta = current_rpe_rot_ - baseline_rpe_rot_;
  bool passed = ate_delta <= ate_regression_threshold_;

  res->passed = passed;
  res->ate_delta = ate_delta;
  res->rpe_trans_delta = rpe_trans_delta;
  res->rpe_rot_delta = rpe_rot_delta;
  res->report_json = std::string("{\"passed\":") + (passed ? "true" : "false") +
    ",\"ate_delta\":" + std::to_string(ate_delta) +
    ",\"rpe_trans_delta\":" + std::to_string(rpe_trans_delta) +
    ",\"rpe_rot_delta\":" + std::to_string(rpe_rot_delta) +
    ",\"threshold\":" + std::to_string(ate_regression_threshold_) + "}";

  RCLCPP_INFO(this->get_logger(),
    "Baseline comparison: %s (ATE delta=%.4f, RPE_t delta=%.4f, threshold=%.4f)",
    passed ? "PASS" : "FAIL", ate_delta, rpe_trans_delta, ate_regression_threshold_);
}

void MapValidator::loadGroundTruth(
  const std::shared_ptr<fastlio2_validation::srv::LoadGroundTruth::Request> req,
  std::shared_ptr<fastlio2_validation::srv::LoadGroundTruth::Response> res)
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (req->format == "TUM" || req->format.empty()) {
    ground_truth_trajectory_ = loadTrajectoryTUM(req->file_path);
  } else {
    res->success = false;
    res->message = "Unsupported format: " + req->format;
    res->num_poses_loaded = 0;
    return;
  }

  has_ground_truth_ = !ground_truth_trajectory_.empty();
  res->success = has_ground_truth_;
  res->num_poses_loaded = static_cast<int>(ground_truth_trajectory_.size());
  res->message = has_ground_truth_ ?
    "Loaded " + std::to_string(res->num_poses_loaded) + " poses" :
    "Failed to load file: " + req->file_path;

  RCLCPP_INFO(this->get_logger(), "Ground truth loaded: %d poses from %s",
              res->num_poses_loaded, req->file_path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// Metrics computation
// ─────────────────────────────────────────────────────────────────────────────

void MapValidator::computeMetrics()
{
  // Always compute RPE (self-referential, no GT needed)
  auto rpe = computeRPE(estimated_trajectory_, delta_m_);
  current_rpe_trans_ = rpe.mean_trans;
  current_rpe_rot_ = rpe.mean_rot_deg;

  // ATE only if ground truth available
  if (has_ground_truth_ && !ground_truth_trajectory_.empty()) {
    current_ate_rmse_ = computeATERMSE(estimated_trajectory_, ground_truth_trajectory_);
  }

  // Publish metrics
  auto ate_msg = std_msgs::msg::Float64();
  ate_msg.data = current_ate_rmse_;
  pub_ate_rmse_->publish(ate_msg);

  auto rpe_t_msg = std_msgs::msg::Float64();
  rpe_t_msg.data = current_rpe_trans_;
  pub_rpe_trans_->publish(rpe_t_msg);

  auto rpe_r_msg = std_msgs::msg::Float64();
  rpe_r_msg.data = current_rpe_rot_;
  pub_rpe_rot_->publish(rpe_r_msg);

  // JSON report
  std::ostringstream ss;
  ss << "{\"ate_rmse\": " << current_ate_rmse_
     << ", \"rpe_trans_mean\": " << current_rpe_trans_
     << ", \"rpe_rot_mean_deg\": " << current_rpe_rot_
     << ", \"has_ground_truth\": " << (has_ground_truth_ ? "true" : "false")
     << ", \"num_estimated_poses\": " << estimated_trajectory_.size()
     << ", \"num_gt_poses\": " << ground_truth_trajectory_.size()
     << "}";
  auto report_msg = std_msgs::msg::String();
  report_msg.data = ss.str();
  pub_report_->publish(report_msg);

  RCLCPP_DEBUG(this->get_logger(),
    "Metrics: ATE_RMSE=%.4f, RPE_t=%.4f, RPE_r=%.2f",
    current_ate_rmse_, current_rpe_trans_, current_rpe_rot_);
}

// ─────────────────────────────────────────────────────────────────────────────
// Horn's closed-form SE3 alignment (for ATE)
// ─────────────────────────────────────────────────────────────────────────────

Eigen::Matrix4d MapValidator::hornAlignment(
  const std::vector<Eigen::Vector3d>& src,
  const std::vector<Eigen::Vector3d>& dst)
{
  size_t n = std::min(src.size(), dst.size());
  if (n < 3) return Eigen::Matrix4d::Identity();

  // Compute centroids
  Eigen::Vector3d centroid_src = Eigen::Vector3d::Zero();
  Eigen::Vector3d centroid_dst = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < n; ++i) {
    centroid_src += src[i];
    centroid_dst += dst[i];
  }
  centroid_src /= static_cast<double>(n);
  centroid_dst /= static_cast<double>(n);

  // Compute H matrix
  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < n; ++i) {
    H += (src[i] - centroid_src) * (dst[i] - centroid_dst).transpose();
  }

  // SVD
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();

  // Ensure proper rotation (det = +1)
  if (R.determinant() < 0) {
    Eigen::Matrix3d V = svd.matrixV();
    V.col(2) *= -1;
    R = V * svd.matrixU().transpose();
  }

  Eigen::Vector3d t = centroid_dst - R * centroid_src;

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = t;
  return T;
}

double MapValidator::computeATE(
  const std::vector<Pose3D>& estimated,
  const std::vector<Pose3D>& ground_truth)
{
  size_t n = std::min(estimated.size(), ground_truth.size());
  if (n < 3) return 0.0;

  std::vector<Eigen::Vector3d> src(n), dst(n);
  for (size_t i = 0; i < n; ++i) {
    src[i] = estimated[i].translation;
    dst[i] = ground_truth[i].translation;
  }

  auto T = hornAlignment(src, dst);
  Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  Eigen::Vector3d t = T.block<3, 1>(0, 3);

  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    Eigen::Vector3d aligned = R * src[i] + t;
    sum += (aligned - dst[i]).norm();
  }
  return sum / static_cast<double>(n);
}

double MapValidator::computeATERMSE(
  const std::vector<Pose3D>& estimated,
  const std::vector<Pose3D>& ground_truth)
{
  size_t n = std::min(estimated.size(), ground_truth.size());
  if (n < 3) return 0.0;

  std::vector<Eigen::Vector3d> src(n), dst(n);
  for (size_t i = 0; i < n; ++i) {
    src[i] = estimated[i].translation;
    dst[i] = ground_truth[i].translation;
  }

  auto T = hornAlignment(src, dst);
  Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  Eigen::Vector3d t = T.block<3, 1>(0, 3);

  double sum_sq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    Eigen::Vector3d aligned = R * src[i] + t;
    sum_sq += (aligned - dst[i]).squaredNorm();
  }
  return std::sqrt(sum_sq / static_cast<double>(n));
}

// ─────────────────────────────────────────────────────────────────────────────
// RPE (self-referential)
// ─────────────────────────────────────────────────────────────────────────────

MapValidator::RPEResult MapValidator::computeRPE(
  const std::vector<Pose3D>& trajectory, double delta_m)
{
  RPEResult result{0.0, 0.0};
  if (trajectory.size() < 2) return result;

  std::vector<double> trans_errors;
  std::vector<double> rot_errors;

  for (size_t i = 0; i < trajectory.size(); ++i) {
    // Find next pose at ~delta_m distance
    double acc_dist = 0.0;
    size_t j = i + 1;
    while (j < trajectory.size()) {
      acc_dist += (trajectory[j].translation - trajectory[j - 1].translation).norm();
      if (acc_dist >= delta_m) break;
      j++;
    }
    if (j >= trajectory.size()) break;

    // Relative pose: T_i^{-1} * T_j
    Eigen::Matrix3d R_i = trajectory[i].rotation.toRotationMatrix();
    Eigen::Matrix3d R_j = trajectory[j].rotation.toRotationMatrix();
    Eigen::Vector3d t_i = trajectory[i].translation;
    Eigen::Vector3d t_j = trajectory[j].translation;

    Eigen::Matrix3d R_rel = R_i.transpose() * R_j;
    Eigen::Vector3d t_rel = R_i.transpose() * (t_j - t_i);

    // Translation error: deviation from expected distance
    double trans_err = std::abs(t_rel.norm() - acc_dist);
    trans_errors.push_back(trans_err);

    // Rotation error: angle of relative rotation deviation
    // In self-referential mode, we compare consecutive sub-sequences
    // The error is the angular magnitude itself (deviation from identity)
    double angle = std::acos(std::clamp(
      (R_rel.trace() - 1.0) / 2.0, -1.0, 1.0));
    rot_errors.push_back(angle * 180.0 / M_PI);
  }

  if (!trans_errors.empty()) {
    result.mean_trans = std::accumulate(trans_errors.begin(), trans_errors.end(), 0.0) /
                        trans_errors.size();
    result.mean_rot_deg = std::accumulate(rot_errors.begin(), rot_errors.end(), 0.0) /
                          rot_errors.size();
  }

  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// File loading
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Pose3D> MapValidator::loadTrajectoryTUM(const std::string& path)
{
  std::vector<Pose3D> result;
  std::ifstream file(path);
  if (!file.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Cannot open trajectory file: %s", path.c_str());
    return result;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    Pose3D p;
    double qx, qy, qz, qw;
    ss >> p.timestamp >> p.translation.x() >> p.translation.y() >> p.translation.z()
       >> qx >> qy >> qz >> qw;
    p.rotation = Eigen::Quaterniond(qw, qx, qy, qz);
    result.push_back(p);
  }

  return result;
}

}  // namespace fastlio2_validation

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fastlio2_validation::MapValidator>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
