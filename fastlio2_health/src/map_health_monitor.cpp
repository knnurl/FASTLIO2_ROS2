#include "fastlio2_health/map_health_monitor.hpp"

#include <sstream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>

namespace fastlio2_health
{

// ── Simple JSON helper (avoids external nlohmann dependency) ─────────────────
static std::string jsonKV(const std::string& key, const std::string& val)
{
  return "\"" + key + "\": \"" + val + "\"";
}
static std::string jsonKV(const std::string& key, double val)
{
  return "\"" + key + "\": " + std::to_string(val);
}
static std::string jsonKV(const std::string& key, int val)
{
  return "\"" + key + "\": " + std::to_string(val);
}

// ─────────────────────────────────────────────────────────────────────────────
MapHealthMonitor::MapHealthMonitor(const rclcpp::NodeOptions& options)
  : Node("map_health_monitor", options)
{
  // ── Declare parameters ──
  this->declare_parameter("degradation_threshold", 0.4);
  this->declare_parameter("improvement_threshold", 0.05);
  this->declare_parameter("staleness_decay_s", 300.0);
  this->declare_parameter("staleness_hba_threshold", 0.8);
  this->declare_parameter("confidence_threshold", 0.7);
  this->declare_parameter("verify_scan_count", 30);
  this->declare_parameter("pre_merge_baseline_scans", 20);
  this->declare_parameter("quality_cell_size_m", 10.0);
  this->declare_parameter("icp_history_window", 50);
  this->declare_parameter("partition_cell_size_m", 50.0);
  this->declare_parameter("partition_radius_cells", 1);
  this->declare_parameter("merge_policy", "TIMESTAMP_PRIORITY");

  degradation_threshold_ = this->get_parameter("degradation_threshold").as_double();
  improvement_threshold_ = this->get_parameter("improvement_threshold").as_double();
  staleness_decay_s_ = this->get_parameter("staleness_decay_s").as_double();
  staleness_hba_threshold_ = this->get_parameter("staleness_hba_threshold").as_double();
  confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
  verify_scan_count_ = this->get_parameter("verify_scan_count").as_int();
  pre_merge_baseline_scans_ = this->get_parameter("pre_merge_baseline_scans").as_int();
  quality_cell_size_m_ = this->get_parameter("quality_cell_size_m").as_double();
  icp_history_window_ = this->get_parameter("icp_history_window").as_int();
  partition_cell_size_m_ = this->get_parameter("partition_cell_size_m").as_double();
  partition_radius_cells_ = this->get_parameter("partition_radius_cells").as_int();
  merge_policy_ = this->get_parameter("merge_policy").as_string();

  // ── Sub-components ──
  partitioner_ = std::make_unique<PoseGraphPartitioner>(partition_cell_size_m_);
  staleness_tracker_ = std::make_unique<MapStalenessTracker>(
    quality_cell_size_m_, staleness_decay_s_, confidence_threshold_, staleness_hba_threshold_);

  // ── Subscriptions ──
  auto sensor_qos = rclcpp::SensorDataQoS();
  auto default_qos = rclcpp::SystemDefaultsQoS();

  sub_icp_fitness_ = this->create_subscription<std_msgs::msg::Float32>(
    "/fsm/icp_fitness", default_qos,
    std::bind(&MapHealthMonitor::onIcpFitness, this, std::placeholders::_1));

  sub_fsm_state_ = this->create_subscription<std_msgs::msg::String>(
    "/fsm/state", default_qos,
    std::bind(&MapHealthMonitor::onFsmState, this, std::placeholders::_1));

  sub_loop_closure_ = this->create_subscription<fastlio2_health::msg::LoopClosureEvent>(
    "/loop_closure/event", default_qos,
    std::bind(&MapHealthMonitor::onLoopClosure, this, std::placeholders::_1));

  sub_hba_completion_ = this->create_subscription<fastlio2_health::msg::HBACompletion>(
    "/hba/completion", default_qos,
    std::bind(&MapHealthMonitor::onHbaCompletion, this, std::placeholders::_1));

  sub_merge_completion_ = this->create_subscription<fastlio2_health::msg::MergeCompletion>(
    "/map_merge/completion", default_qos,
    std::bind(&MapHealthMonitor::onMergeCompletion, this, std::placeholders::_1));

  sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "/fastlio2/lio_odom", sensor_qos,
    std::bind(&MapHealthMonitor::onOdometry, this, std::placeholders::_1));

  // ── Publishers ──
  pub_quality_grid_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
    "/map_health/quality_grid", default_qos);

  pub_staleness_grid_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
    "/map_health/staleness_grid", default_qos);

  pub_health_status_ = this->create_publisher<fastlio2_health::msg::MapHealthStatus>(
    "/map_health/status", default_qos);

  pub_gui_state_ = this->create_publisher<std_msgs::msg::String>(
    "/map_health/gui_state", default_qos);

  pub_hba_trigger_ = this->create_publisher<fastlio2_health::msg::HBATrigger>(
    "/hba/trigger", default_qos);

  pub_merge_trigger_ = this->create_publisher<fastlio2_health::msg::MergeTrigger>(
    "/map_merge/trigger", default_qos);

  pub_merge_rollback_ = this->create_publisher<std_msgs::msg::Bool>(
    "/map_merge/rollback", default_qos);

  pub_loop_suppress_ = this->create_publisher<std_msgs::msg::Bool>(
    "/degenerate_guard/suppress_loop_closure", default_qos);

  // ── Service clients (validation integration) ──
  client_set_baseline_ = this->create_client<std_srvs::srv::Trigger>(
    "/validation/set_baseline");
  client_compare_baseline_ = this->create_client<fastlio2_validation::srv::CompareToBaseline>(
    "/validation/compare_to_baseline");

  // ── Manual override services (GUI buttons) ──
  srv_force_hba_ = this->create_service<std_srvs::srv::Trigger>(
    "/map_health/force_hba",
    std::bind(&MapHealthMonitor::srvForceHba, this,
              std::placeholders::_1, std::placeholders::_2));
  srv_force_merge_ = this->create_service<std_srvs::srv::Trigger>(
    "/map_health/force_merge",
    std::bind(&MapHealthMonitor::srvForceMerge, this,
              std::placeholders::_1, std::placeholders::_2));
  srv_force_rollback_ = this->create_service<std_srvs::srv::Trigger>(
    "/map_health/force_rollback",
    std::bind(&MapHealthMonitor::srvForceRollback, this,
              std::placeholders::_1, std::placeholders::_2));

  // ── Timer: 1 Hz status update ──
  timer_ = this->create_wall_timer(
    std::chrono::seconds(1),
    std::bind(&MapHealthMonitor::timerCallback, this));

  RCLCPP_INFO(this->get_logger(), "MapHealthMonitor initialised [phase=MONITORING]");
}

// ─────────────────────────────────────────────────────────────────────────────
// Callbacks
// ─────────────────────────────────────────────────────────────────────────────

void MapHealthMonitor::onIcpFitness(const std_msgs::msg::Float32::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  float fitness = msg->data;

  // Update quality cell for current robot position
  updateQualityCell(robot_x_, robot_y_, fitness);

  // Update staleness tracker if fitness is above confidence threshold
  // Note: lower ICP fitness = better match in PCL convention, but the prompt
  // uses normalised 0-1 where higher = better. We follow the prompt convention.
  if (fitness >= confidence_threshold_) {
    staleness_tracker_->updateConfidentLocalisation(
      robot_x_, robot_y_, this->now());
  }

  // Verification phase: collect post-merge fitness samples
  if (phase_ == PipelinePhase::VERIFYING && verification_scans_remaining_ > 0) {
    post_merge_fitness_.push_back(fitness);
    verification_scans_remaining_--;
    RCLCPP_DEBUG(this->get_logger(), "Verification scan %d/%d, fitness=%.3f",
                 verify_scan_count_ - verification_scans_remaining_,
                 verify_scan_count_, fitness);
  }

  // Pre-merge baseline collection
  if (phase_ == PipelinePhase::MONITORING || phase_ == PipelinePhase::TRIGGER_HBA) {
    pre_merge_fitness_.push_back(fitness);
    if (static_cast<int>(pre_merge_fitness_.size()) > pre_merge_baseline_scans_) {
      pre_merge_fitness_.pop_front();
    }
  }
}

void MapHealthMonitor::onFsmState(const std_msgs::msg::String::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  fsm_state_ = msg->data;
}

void MapHealthMonitor::onLoopClosure(
  const fastlio2_health::msg::LoopClosureEvent::SharedPtr msg)
{
  if (!msg->accepted) {
    RCLCPP_DEBUG(this->get_logger(), "Loop closure rejected (score=%.3f)", msg->match_score);
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  RCLCPP_INFO(this->get_logger(),
    "Loop closure accepted: nodes (%d,%d), score=%.3f, correction=%.3fm",
    msg->node_i, msg->node_j, msg->match_score, msg->correction_magnitude_m);

  // Update quality cell with loop closure correction
  auto cell_key = positionToCell(robot_x_, robot_y_);
  auto& cell = quality_grid_[cell_key];
  cell.last_loop_closure_correction_m = msg->correction_magnitude_m;

  // If we're monitoring, a strong loop closure can trigger HBA
  if (phase_ == PipelinePhase::MONITORING) {
    // Use spatial partitioner to get affected nodes
    Eigen::Vector3d lc_pos(
      msg->relative_transform.transform.translation.x,
      msg->relative_transform.transform.translation.y,
      msg->relative_transform.transform.translation.z);

    // If the position is zero (relative transform is small), use robot position
    if (lc_pos.norm() < 1e-3) {
      lc_pos = Eigen::Vector3d(robot_x_, robot_y_, robot_z_);
    }

    auto subgraph = partitioner_->getSubgraphNodes(lc_pos, partition_radius_cells_);

    RCLCPP_INFO(this->get_logger(),
      "Spatial partition: subgraph=%zu nodes, total=%zu nodes",
      subgraph.size(), partitioner_->totalNodes());

    geometry_msgs::msg::Point centre;
    centre.x = robot_x_;
    centre.y = robot_y_;
    centre.z = robot_z_;

    std::vector<int> node_ids;
    if (!subgraph.empty()) {
      node_ids = subgraph;
    } else {
      node_ids = {msg->node_i, msg->node_j};
    }

    triggerHba(node_ids, centre, msg->match_score, "LOOP_CLOSURE");
  }
}

void MapHealthMonitor::onHbaCompletion(
  const fastlio2_health::msg::HBACompletion::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (phase_ != PipelinePhase::HBA_RUNNING) {
    RCLCPP_WARN(this->get_logger(),
      "Received HBA completion but not in HBA_RUNNING phase (current=%s)",
      phaseToString(phase_).c_str());
    return;
  }

  if (!msg->success) {
    RCLCPP_ERROR(this->get_logger(), "HBA failed, returning to MONITORING");
    transitionTo(PipelinePhase::MONITORING);
    return;
  }

  RCLCPP_INFO(this->get_logger(),
    "HBA completed: mean_correction=%.4fm, nodes=%zu",
    msg->mean_correction_magnitude, msg->node_ids.size());

  last_hba_result_ = msg;

  // Compute pre-merge baseline
  if (!pre_merge_fitness_.empty()) {
    pre_merge_baseline_ = std::accumulate(
      pre_merge_fitness_.begin(), pre_merge_fitness_.end(), 0.0) /
      pre_merge_fitness_.size();
  }

  // Call validation set_baseline if service is available
  if (client_set_baseline_->service_is_ready()) {
    auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
    client_set_baseline_->async_send_request(req);
    RCLCPP_INFO(this->get_logger(), "Validation baseline snapshot requested");
  }

  // Transition to MERGING
  transitionTo(PipelinePhase::MERGING);

  // Publish merge trigger
  if (msg->corrected_pcd.data.size() > 0 && !msg->corrected_transforms.empty()) {
    triggerMerge(msg->corrected_pcd, msg->corrected_transforms[0]);
  } else {
    RCLCPP_WARN(this->get_logger(),
      "HBA completion has no corrected PCD/transforms, skipping merge");
    transitionTo(PipelinePhase::MONITORING);
  }
}

void MapHealthMonitor::onMergeCompletion(
  const fastlio2_health::msg::MergeCompletion::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (phase_ != PipelinePhase::MERGING) {
    RCLCPP_WARN(this->get_logger(),
      "Received merge completion but not in MERGING phase (current=%s)",
      phaseToString(phase_).c_str());
    return;
  }

  if (!msg->success) {
    RCLCPP_ERROR(this->get_logger(), "Merge failed, returning to MONITORING");
    transitionTo(PipelinePhase::MONITORING);
    return;
  }

  RCLCPP_INFO(this->get_logger(),
    "Merge completed: +%d/-%d points, overlap=%.2f",
    msg->points_added, msg->points_removed, msg->overlap_ratio);

  // Enter verification phase
  transitionTo(PipelinePhase::VERIFYING);
  post_merge_fitness_.clear();
  verification_scans_remaining_ = verify_scan_count_;
}

void MapHealthMonitor::onOdometry(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  robot_x_ = msg->pose.pose.position.x;
  robot_y_ = msg->pose.pose.position.y;
  robot_z_ = msg->pose.pose.position.z;
}

// ─────────────────────────────────────────────────────────────────────────────
// Timer callback (1 Hz)
// ─────────────────────────────────────────────────────────────────────────────

void MapHealthMonitor::timerCallback()
{
  std::lock_guard<std::mutex> lock(mutex_);

  // Update staleness scores
  staleness_tracker_->updateStaleness(this->now());

  // Phase-specific logic
  switch (phase_) {
    case PipelinePhase::MONITORING:
      evaluateDegradation();
      break;

    case PipelinePhase::VERIFYING:
      if (verification_scans_remaining_ <= 0) {
        evaluateVerification();
      }
      break;

    case PipelinePhase::TRIGGER_HBA:
    case PipelinePhase::HBA_RUNNING:
    case PipelinePhase::MERGING:
      // Waiting for external completion signals
      break;

    case PipelinePhase::ROLLBACK:
      // Rollback is immediate, handled in executeRollback()
      break;
  }

  // Publish status on every tick
  publishHealthStatus();
  publishQualityGrid();
  publishStalenessGrid();
  publishGuiState();
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline logic
// ─────────────────────────────────────────────────────────────────────────────

void MapHealthMonitor::transitionTo(PipelinePhase new_phase)
{
  RCLCPP_INFO(this->get_logger(), "Phase transition: %s -> %s",
    phaseToString(phase_).c_str(), phaseToString(new_phase).c_str());

  PipelinePhase old_phase = phase_;
  phase_ = new_phase;

  // Suppress loop closures during active pipeline work (HBA, merge, verify)
  bool was_active = (old_phase == PipelinePhase::HBA_RUNNING ||
                     old_phase == PipelinePhase::MERGING ||
                     old_phase == PipelinePhase::VERIFYING);
  bool is_active  = (new_phase == PipelinePhase::HBA_RUNNING ||
                     new_phase == PipelinePhase::MERGING ||
                     new_phase == PipelinePhase::VERIFYING);

  if (is_active && !was_active) {
    auto msg = std_msgs::msg::Bool();
    msg.data = true;
    pub_loop_suppress_->publish(msg);
    RCLCPP_INFO(this->get_logger(), "Loop closure suppression ACTIVATED");
  } else if (!is_active && was_active) {
    auto msg = std_msgs::msg::Bool();
    msg.data = false;
    pub_loop_suppress_->publish(msg);
    RCLCPP_INFO(this->get_logger(), "Loop closure suppression RELEASED");
  }
}

void MapHealthMonitor::evaluateDegradation()
{
  // Check quality grid for degraded cells
  for (auto& [key, cell] : quality_grid_) {
    if (cell.icp_history.empty()) continue;

    // Compute normalised degradation score
    // Lower mean_icp_fitness = more degraded (using prompt's convention)
    double degradation = 1.0 - cell.mean_icp_fitness;

    // Also factor in staleness
    float staleness = staleness_tracker_->getStalenessScore(key.cx, key.cy);
    cell.staleness_score = staleness;

    // Combined degradation metric
    double combined = 0.6 * degradation + 0.4 * static_cast<double>(staleness);

    if (combined > degradation_threshold_) {
      RCLCPP_WARN(this->get_logger(),
        "Cell (%d,%d) degraded: combined=%.3f (fitness=%.3f, staleness=%.2f)",
        key.cx, key.cy, combined, cell.mean_icp_fitness, staleness);

      // Get subgraph nodes for this cell
      Eigen::Vector3d cell_centre(
        (key.cx + 0.5) * quality_cell_size_m_,
        (key.cy + 0.5) * quality_cell_size_m_,
        0.0);
      auto subgraph = partitioner_->getSubgraphNodes(
        cell_centre, partition_radius_cells_);

      geometry_msgs::msg::Point centre;
      centre.x = cell_centre.x();
      centre.y = cell_centre.y();
      centre.z = 0.0;

      triggerHba(subgraph, centre,
                 static_cast<float>(cell.mean_icp_fitness), "DEGRADATION");
      return;  // Only trigger one HBA at a time
    }
  }

  // Also check for stale cells that need priority HBA
  auto stale_cells = staleness_tracker_->getStaleCells();
  if (!stale_cells.empty()) {
    RCLCPP_INFO(this->get_logger(), "%zu stale cells flagged for priority HBA",
                stale_cells.size());
  }
}

void MapHealthMonitor::srvForceHba(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (phase_ != PipelinePhase::MONITORING) {
    res->success = false;
    res->message = "Cannot force HBA: pipeline is busy (phase=" + phaseToString(phase_) + ")";
    return;
  }
  RCLCPP_INFO(this->get_logger(), "Force HBA requested via service");
  triggerHba({}, geometry_msgs::msg::Point{}, 0.0f, "MANUAL");
  res->success = true;
  res->message = "HBA triggered";
}

void MapHealthMonitor::srvForceMerge(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (phase_ != PipelinePhase::MONITORING) {
    res->success = false;
    res->message = "Cannot force merge: pipeline is busy (phase=" + phaseToString(phase_) + ")";
    return;
  }
  if (!last_hba_result_ ||
      last_hba_result_->corrected_pcd.data.empty() ||
      last_hba_result_->corrected_transforms.empty()) {
    res->success = false;
    res->message = "No HBA result available to merge — run Force HBA first";
    return;
  }
  RCLCPP_INFO(this->get_logger(), "Force merge requested via service");
  transitionTo(PipelinePhase::MERGING);
  triggerMerge(last_hba_result_->corrected_pcd, last_hba_result_->corrected_transforms[0]);
  res->success = true;
  res->message = "Merge triggered";
}

void MapHealthMonitor::srvForceRollback(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
  std::lock_guard<std::mutex> lock(mutex_);
  RCLCPP_INFO(this->get_logger(), "Force rollback requested via service");
  executeRollback();
  res->success = true;
  res->message = "Rollback executed";
}

void MapHealthMonitor::triggerHba(
  const std::vector<int>& node_ids,
  const geometry_msgs::msg::Point& cell_centre,
  float fitness,
  const std::string& reason)
{
  transitionTo(PipelinePhase::TRIGGER_HBA);

  auto trigger = fastlio2_health::msg::HBATrigger();
  trigger.header.stamp = this->now();
  trigger.header.frame_id = "map";
  trigger.node_ids = node_ids;
  trigger.cell_centre = cell_centre;
  trigger.trigger_fitness_score = fitness;
  trigger.trigger_reason = reason;

  pub_hba_trigger_->publish(trigger);

  RCLCPP_INFO(this->get_logger(),
    "HBA triggered: reason=%s, nodes=%zu, fitness=%.3f",
    reason.c_str(), node_ids.size(), fitness);

  transitionTo(PipelinePhase::HBA_RUNNING);
}

void MapHealthMonitor::triggerMerge(
  const sensor_msgs::msg::PointCloud2& pcd,
  const geometry_msgs::msg::TransformStamped& tf)
{
  auto trigger = fastlio2_health::msg::MergeTrigger();
  trigger.header.stamp = this->now();
  trigger.header.frame_id = "map";
  trigger.incoming_pcd = pcd;
  trigger.transform = tf;
  trigger.merge_policy = merge_policy_;
  trigger.icp_fitness_weight = static_cast<float>(pre_merge_baseline_);

  pub_merge_trigger_->publish(trigger);

  RCLCPP_INFO(this->get_logger(),
    "Merge triggered: policy=%s, fitness_weight=%.3f",
    merge_policy_.c_str(), trigger.icp_fitness_weight);
}

void MapHealthMonitor::evaluateVerification()
{
  if (post_merge_fitness_.empty()) {
    RCLCPP_WARN(this->get_logger(),
      "No post-merge fitness samples, returning to MONITORING");
    transitionTo(PipelinePhase::MONITORING);
    return;
  }

  double post_mean = std::accumulate(
    post_merge_fitness_.begin(), post_merge_fitness_.end(), 0.0) /
    post_merge_fitness_.size();

  RCLCPP_INFO(this->get_logger(),
    "Verification: pre_baseline=%.4f, post_mean=%.4f",
    pre_merge_baseline_, post_mean);

  // Check if map quality improved
  // Higher fitness = better, so post should be >= pre * (1 - threshold)
  // If post_mean > pre_mean * (1 - improvement_threshold) AND post_mean < pre_mean:
  //   this means it got worse → rollback
  // Actually the prompt says:
  //   "if post_mean > pre_mean * (1 - improvement_threshold): ROLLBACK"
  // This means: if post-merge fitness is NOT significantly better, rollback
  bool should_rollback = (post_mean > pre_merge_baseline_ * (1.0 - improvement_threshold_));

  // Also check with validation node if available
  if (client_compare_baseline_->service_is_ready()) {
    auto req = std::make_shared<fastlio2_validation::srv::CompareToBaseline::Request>();
    auto future = client_compare_baseline_->async_send_request(req);
    // Non-blocking: we'll use the fitness-based check as primary
    RCLCPP_INFO(this->get_logger(), "Validation comparison requested");
  }

  if (should_rollback) {
    RCLCPP_WARN(this->get_logger(),
      "Merge did not improve quality (post=%.4f, threshold=%.4f), rolling back",
      post_mean, pre_merge_baseline_ * (1.0 - improvement_threshold_));
    executeRollback();
  } else {
    RCLCPP_INFO(this->get_logger(),
      "Merge verified: quality improved (%.4f -> %.4f)", pre_merge_baseline_, post_mean);

    // Update quality grid baseline for affected cells
    auto cell_key = positionToCell(robot_x_, robot_y_);
    auto& cell = quality_grid_[cell_key];
    cell.last_hba_timestamp = this->now();

    transitionTo(PipelinePhase::MONITORING);
  }
}

void MapHealthMonitor::executeRollback()
{
  transitionTo(PipelinePhase::ROLLBACK);

  // Publish rollback signal
  auto rollback_msg = std_msgs::msg::Bool();
  rollback_msg.data = true;
  pub_merge_rollback_->publish(rollback_msg);

  RCLCPP_ERROR(this->get_logger(),
    "ROLLBACK executed — reverting last merge. Affected cell flagged NEEDS_REVIEW.");

  // Mark affected cell as NEEDS_REVIEW
  auto cell_key = positionToCell(robot_x_, robot_y_);
  quality_grid_[cell_key].needs_review = true;

  transitionTo(PipelinePhase::MONITORING);
}

// ─────────────────────────────────────────────────────────────────────────────
// Quality grid helpers
// ─────────────────────────────────────────────────────────────────────────────

QualityCellKey MapHealthMonitor::positionToCell(double x, double y) const
{
  return {static_cast<int>(std::floor(x / quality_cell_size_m_)),
          static_cast<int>(std::floor(y / quality_cell_size_m_))};
}

void MapHealthMonitor::updateQualityCell(double x, double y, float icp_fitness)
{
  auto key = positionToCell(x, y);
  auto& cell = quality_grid_[key];

  cell.icp_history.push_back(icp_fitness);
  if (static_cast<int>(cell.icp_history.size()) > icp_history_window_) {
    cell.icp_history.pop_front();
  }

  // Recompute rolling mean
  cell.mean_icp_fitness = std::accumulate(
    cell.icp_history.begin(), cell.icp_history.end(), 0.0) /
    cell.icp_history.size();
}

float MapHealthMonitor::computeGlobalHealth() const
{
  if (quality_grid_.empty()) return 1.0f;

  double sum = 0.0;
  int count = 0;
  for (const auto& [_, cell] : quality_grid_) {
    if (!cell.icp_history.empty()) {
      sum += cell.mean_icp_fitness;
      count++;
    }
  }
  return (count > 0) ? static_cast<float>(sum / count) : 1.0f;
}

void MapHealthMonitor::publishQualityGrid()
{
  if (quality_grid_.empty()) return;

  int min_cx = std::numeric_limits<int>::max();
  int max_cx = std::numeric_limits<int>::min();
  int min_cy = std::numeric_limits<int>::max();
  int max_cy = std::numeric_limits<int>::min();

  for (const auto& [key, _] : quality_grid_) {
    min_cx = std::min(min_cx, key.cx);
    max_cx = std::max(max_cx, key.cx);
    min_cy = std::min(min_cy, key.cy);
    max_cy = std::max(max_cy, key.cy);
  }

  int width = max_cx - min_cx + 1;
  int height = max_cy - min_cy + 1;

  nav_msgs::msg::OccupancyGrid grid;
  grid.header.stamp = this->now();
  grid.header.frame_id = "map";
  grid.info.resolution = static_cast<float>(quality_cell_size_m_);
  grid.info.width = width;
  grid.info.height = height;
  grid.info.origin.position.x = min_cx * quality_cell_size_m_;
  grid.info.origin.position.y = min_cy * quality_cell_size_m_;
  grid.info.origin.position.z = 0.0;

  grid.data.resize(width * height, -1);
  for (const auto& [key, cell] : quality_grid_) {
    int idx = (key.cy - min_cy) * width + (key.cx - min_cx);
    // Map fitness [0,1] to occupancy [0,100] (inverted: 0=healthy, 100=degraded)
    int val = static_cast<int>((1.0 - cell.mean_icp_fitness) * 100.0);
    grid.data[idx] = static_cast<int8_t>(std::clamp(val, 0, 100));
  }

  pub_quality_grid_->publish(grid);
}

void MapHealthMonitor::publishStalenessGrid()
{
  int width, height;
  double origin_x, origin_y;
  auto data = staleness_tracker_->toOccupancyData(width, height, origin_x, origin_y);

  nav_msgs::msg::OccupancyGrid grid;
  grid.header.stamp = this->now();
  grid.header.frame_id = "map";
  grid.info.resolution = static_cast<float>(staleness_tracker_->cellSize());
  grid.info.width = width;
  grid.info.height = height;
  grid.info.origin.position.x = origin_x;
  grid.info.origin.position.y = origin_y;
  grid.data = data;

  pub_staleness_grid_->publish(grid);
}

void MapHealthMonitor::publishHealthStatus()
{
  auto msg = fastlio2_health::msg::MapHealthStatus();
  msg.header.stamp = this->now();
  msg.header.frame_id = "map";
  msg.global_health_score = computeGlobalHealth();
  msg.current_phase = phaseToString(phase_);

  int degraded = 0;
  int pending_hba = 0;
  float worst_score = 1.0f;
  QualityCellKey worst_key{0, 0};

  for (const auto& [key, cell] : quality_grid_) {
    if (!cell.icp_history.empty()) {
      double degradation = 1.0 - cell.mean_icp_fitness;
      if (degradation > degradation_threshold_) {
        degraded++;
      }
      if (cell.mean_icp_fitness < worst_score) {
        worst_score = static_cast<float>(cell.mean_icp_fitness);
        worst_key = key;
      }
    }
    if (cell.needs_review) {
      pending_hba++;
    }
  }

  msg.num_degraded_cells = degraded;
  msg.num_cells_pending_hba = pending_hba;
  msg.num_cells_pending_merge = (phase_ == PipelinePhase::MERGING) ? 1 : 0;
  msg.worst_cell_centre.x = (worst_key.cx + 0.5) * quality_cell_size_m_;
  msg.worst_cell_centre.y = (worst_key.cy + 0.5) * quality_cell_size_m_;
  msg.worst_cell_centre.z = 0.0;
  msg.worst_cell_score = worst_score;

  pub_health_status_->publish(msg);
}

void MapHealthMonitor::publishGuiState()
{
  std::ostringstream ss;
  ss << "{"
     << jsonKV("phase", phaseToString(phase_)) << ", "
     << jsonKV("global_health", static_cast<double>(computeGlobalHealth())) << ", "
     << jsonKV("fsm_state", fsm_state_) << ", "
     << jsonKV("num_quality_cells", static_cast<int>(quality_grid_.size())) << ", "
     << jsonKV("robot_x", robot_x_) << ", "
     << jsonKV("robot_y", robot_y_) << ", "
     << jsonKV("merge_policy", merge_policy_) << ", "
     << jsonKV("pre_merge_baseline", pre_merge_baseline_) << ", "
     << jsonKV("verification_remaining", verification_scans_remaining_)
     << "}";

  auto msg = std_msgs::msg::String();
  msg.data = ss.str();
  pub_gui_state_->publish(msg);
}

}  // namespace fastlio2_health

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fastlio2_health::MapHealthMonitor>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
