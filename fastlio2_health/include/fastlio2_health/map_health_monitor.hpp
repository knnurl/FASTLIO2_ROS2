#pragma once

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "fastlio2_health/msg/map_health_status.hpp"
#include "fastlio2_health/msg/hba_trigger.hpp"
#include "fastlio2_health/msg/hba_completion.hpp"
#include "fastlio2_health/msg/merge_trigger.hpp"
#include "fastlio2_health/msg/merge_completion.hpp"
#include "fastlio2_health/msg/loop_closure_event.hpp"

#include "fastlio2_health/pose_graph_partitioner.hpp"
#include "fastlio2_health/map_staleness_tracker.hpp"
#include "fastlio2_validation/srv/compare_to_baseline.hpp"

#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <cmath>

namespace fastlio2_health
{

// ── Spatial quality cell ─────────────────────────────────────────────────────
struct QualityCell
{
  double mean_icp_fitness = 0.0;
  double last_loop_closure_correction_m = 0.0;
  rclcpp::Time last_hba_timestamp{0, 0, RCL_ROS_TIME};
  float staleness_score = 0.0f;
  std::deque<float> icp_history;
  bool needs_review = false;
};

struct QualityCellKey
{
  int cx, cy;
  bool operator==(const QualityCellKey& o) const { return cx == o.cx && cy == o.cy; }
};

struct QualityCellKeyHash
{
  size_t operator()(const QualityCellKey& k) const
  {
    return std::hash<int>{}(k.cx) ^ (std::hash<int>{}(k.cy) << 16);
  }
};

// ── Pipeline phase ───────────────────────────────────────────────────────────
enum class PipelinePhase
{
  MONITORING,
  TRIGGER_HBA,
  HBA_RUNNING,
  MERGING,
  VERIFYING,
  ROLLBACK
};

inline std::string phaseToString(PipelinePhase p)
{
  switch (p) {
    case PipelinePhase::MONITORING:   return "MONITORING";
    case PipelinePhase::TRIGGER_HBA:  return "TRIGGER_HBA";
    case PipelinePhase::HBA_RUNNING:  return "HBA_RUNNING";
    case PipelinePhase::MERGING:      return "MERGING";
    case PipelinePhase::VERIFYING:    return "VERIFYING";
    case PipelinePhase::ROLLBACK:     return "ROLLBACK";
  }
  return "UNKNOWN";
}

// ── Main node ────────────────────────────────────────────────────────────────
class MapHealthMonitor : public rclcpp::Node
{
public:
  explicit MapHealthMonitor(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  // ── Parameters ──
  double degradation_threshold_;
  double improvement_threshold_;
  double staleness_decay_s_;
  double staleness_hba_threshold_;
  double confidence_threshold_;
  int verify_scan_count_;
  int pre_merge_baseline_scans_;
  double quality_cell_size_m_;
  int icp_history_window_;
  double partition_cell_size_m_;
  int partition_radius_cells_;
  std::string merge_policy_;

  // ── Callbacks ──
  void onIcpFitness(const std_msgs::msg::Float32::SharedPtr msg);
  void onFsmState(const std_msgs::msg::String::SharedPtr msg);
  void onLoopClosure(const fastlio2_health::msg::LoopClosureEvent::SharedPtr msg);
  void onHbaCompletion(const fastlio2_health::msg::HBACompletion::SharedPtr msg);
  void onMergeCompletion(const fastlio2_health::msg::MergeCompletion::SharedPtr msg);
  void onOdometry(const nav_msgs::msg::Odometry::SharedPtr msg);

  // ── Timer ──
  void timerCallback();

  // ── Pipeline logic ──
  void transitionTo(PipelinePhase new_phase);
  void evaluateDegradation();
  void srvForceHba(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                   std::shared_ptr<std_srvs::srv::Trigger::Response> res);
  void srvForceMerge(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                     std::shared_ptr<std_srvs::srv::Trigger::Response> res);
  void srvForceRollback(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                        std::shared_ptr<std_srvs::srv::Trigger::Response> res);

  void triggerHba(const std::vector<int>& node_ids,
                  const geometry_msgs::msg::Point& cell_centre,
                  float fitness, const std::string& reason);
  void triggerMerge(const sensor_msgs::msg::PointCloud2& pcd,
                    const geometry_msgs::msg::TransformStamped& tf);
  void evaluateVerification();
  void executeRollback();

  // ── Quality grid helpers ──
  QualityCellKey positionToCell(double x, double y) const;
  void updateQualityCell(double x, double y, float icp_fitness);
  float computeGlobalHealth() const;
  void publishQualityGrid();
  void publishStalenessGrid();
  void publishHealthStatus();
  void publishGuiState();

  // ── State ──
  PipelinePhase phase_ = PipelinePhase::MONITORING;
  std::mutex mutex_;

  // Current robot position
  double robot_x_ = 0.0;
  double robot_y_ = 0.0;
  double robot_z_ = 0.0;
  std::string fsm_state_ = "UNKNOWN";

  // Spatial quality grid
  std::unordered_map<QualityCellKey, QualityCell, QualityCellKeyHash> quality_grid_;

  // Verification state
  std::deque<float> pre_merge_fitness_;
  std::deque<float> post_merge_fitness_;
  double pre_merge_baseline_ = 0.0;
  int verification_scans_remaining_ = 0;

  // Last HBA completion data (for merge trigger)
  fastlio2_health::msg::HBACompletion::SharedPtr last_hba_result_;

  // ── Sub-components ──
  std::unique_ptr<PoseGraphPartitioner> partitioner_;
  std::unique_ptr<MapStalenessTracker> staleness_tracker_;

  // ── Subscriptions ──
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_icp_fitness_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_fsm_state_;
  rclcpp::Subscription<fastlio2_health::msg::LoopClosureEvent>::SharedPtr sub_loop_closure_;
  rclcpp::Subscription<fastlio2_health::msg::HBACompletion>::SharedPtr sub_hba_completion_;
  rclcpp::Subscription<fastlio2_health::msg::MergeCompletion>::SharedPtr sub_merge_completion_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;

  // ── Publishers ──
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_quality_grid_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pub_staleness_grid_;
  rclcpp::Publisher<fastlio2_health::msg::MapHealthStatus>::SharedPtr pub_health_status_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_gui_state_;
  rclcpp::Publisher<fastlio2_health::msg::HBATrigger>::SharedPtr pub_hba_trigger_;
  rclcpp::Publisher<fastlio2_health::msg::MergeTrigger>::SharedPtr pub_merge_trigger_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_merge_rollback_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_loop_suppress_;

  // ── Timer ──
  rclcpp::TimerBase::SharedPtr timer_;

  // ── Service client for validation baseline ──
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr client_set_baseline_;
  rclcpp::Client<fastlio2_validation::srv::CompareToBaseline>::SharedPtr client_compare_baseline_;

  // Manual override services (called by the GUI)
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_force_hba_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_force_merge_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_force_rollback_;
};

}  // namespace fastlio2_health
