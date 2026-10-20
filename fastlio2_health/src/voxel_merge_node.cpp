#include "fastlio2_health/voxel_merge_node.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <Eigen/Geometry>

#include <unordered_map>
#include <random>

namespace fastlio2_health
{

// ── Voxel key for spatial hashing ────────────────────────────────────────────
struct VoxelKey
{
  int x, y, z;
  bool operator==(const VoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct VoxelKeyHash
{
  size_t operator()(const VoxelKey& k) const
  {
    size_t h = std::hash<int>{}(k.x);
    h ^= std::hash<int>{}(k.y) << 11;
    h ^= std::hash<int>{}(k.z) << 22;
    return h;
  }
};

static VoxelKey pointToVoxel(const PointType& pt, float res)
{
  return {static_cast<int>(std::floor(pt.x / res)),
          static_cast<int>(std::floor(pt.y / res)),
          static_cast<int>(std::floor(pt.z / res))};
}

using VoxelMap = std::unordered_map<VoxelKey, std::vector<PointType>, VoxelKeyHash>;

static VoxelMap buildVoxelMap(const CloudType::Ptr& cloud, float res)
{
  VoxelMap map;
  for (const auto& pt : cloud->points) {
    map[pointToVoxel(pt, res)].push_back(pt);
  }
  return map;
}

// ─────────────────────────────────────────────────────────────────────────────
VoxelMergeNode::VoxelMergeNode(const rclcpp::NodeOptions& options)
  : Node("voxel_merge_node", options)
{
  this->declare_parameter("merge_policy", "TIMESTAMP_PRIORITY");
  this->declare_parameter("voxel_resolution", 0.3);

  merge_policy_ = this->get_parameter("merge_policy").as_string();
  voxel_resolution_ = static_cast<float>(this->get_parameter("voxel_resolution").as_double());

  auto default_qos = rclcpp::SystemDefaultsQoS();
  auto sensor_qos = rclcpp::SensorDataQoS();

  sub_trigger_ = this->create_subscription<fastlio2_health::msg::MergeTrigger>(
    "/map_merge/trigger", default_qos,
    std::bind(&VoxelMergeNode::onMergeTrigger, this, std::placeholders::_1));

  sub_rollback_ = this->create_subscription<std_msgs::msg::Bool>(
    "/map_merge/rollback", default_qos,
    std::bind(&VoxelMergeNode::onRollback, this, std::placeholders::_1));

  sub_current_map_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/fastlio2/world_cloud", sensor_qos,
    std::bind(&VoxelMergeNode::onCurrentMap, this, std::placeholders::_1));

  pub_completion_ = this->create_publisher<fastlio2_health::msg::MergeCompletion>(
    "/map_merge/completion", default_qos);

  pub_merged_map_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/map_merge/merged_cloud", sensor_qos);

  // Runtime-reconfigurable merge_policy parameter
  auto param_cb = [this](const std::vector<rclcpp::Parameter>& params) {
    auto result = rcl_interfaces::msg::SetParametersResult();
    result.successful = true;
    for (const auto& p : params) {
      if (p.get_name() == "merge_policy") {
        auto val = p.as_string();
        if (val == "TIMESTAMP_PRIORITY" || val == "DENSITY_PRIORITY" || val == "CONFIDENCE_WEIGHTED") {
          merge_policy_ = val;
          RCLCPP_INFO(this->get_logger(), "Merge policy changed to: %s", val.c_str());
        } else {
          result.successful = false;
          result.reason = "Invalid merge policy: " + val;
        }
      }
    }
    return result;
  };
  this->add_on_set_parameters_callback(param_cb);

  RCLCPP_INFO(this->get_logger(), "VoxelMergeNode initialised [policy=%s, voxel_res=%.2f]",
              merge_policy_.c_str(), voxel_resolution_);
}

void VoxelMergeNode::onCurrentMap(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!current_map_) {
    current_map_ = std::make_shared<CloudType>();
  }
  pcl::fromROSMsg(*msg, *current_map_);
}

void VoxelMergeNode::onMergeTrigger(const fastlio2_health::msg::MergeTrigger::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (!current_map_ || current_map_->empty()) {
    RCLCPP_WARN(this->get_logger(), "No current map available for merge");
    auto comp = fastlio2_health::msg::MergeCompletion();
    comp.header.stamp = this->now();
    comp.success = false;
    pub_completion_->publish(comp);
    return;
  }

  // Convert incoming PCD
  auto incoming = std::make_shared<CloudType>();
  pcl::fromROSMsg(msg->incoming_pcd, *incoming);

  // Apply transform to incoming
  const auto& tf = msg->transform.transform;
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  Eigen::Quaternionf q(
    static_cast<float>(tf.rotation.w),
    static_cast<float>(tf.rotation.x),
    static_cast<float>(tf.rotation.y),
    static_cast<float>(tf.rotation.z));
  transform.rotate(q);
  transform.translation() << static_cast<float>(tf.translation.x),
                              static_cast<float>(tf.translation.y),
                              static_cast<float>(tf.translation.z);

  auto transformed_incoming = std::make_shared<CloudType>();
  pcl::transformPointCloud(*incoming, *transformed_incoming, transform);

  // Backup current map for rollback
  backup_map_ = std::make_shared<CloudType>(*current_map_);

  // Select policy (prefer message field, fall back to parameter)
  std::string policy = msg->merge_policy.empty() ? merge_policy_ : msg->merge_policy;

  RCLCPP_INFO(this->get_logger(), "Merge triggered: policy=%s, incoming=%zu pts, existing=%zu pts",
              policy.c_str(), transformed_incoming->size(), current_map_->size());

  fastlio2_health::msg::MergeCompletion result;
  if (policy == "DENSITY_PRIORITY") {
    result = mergeDensityPriority(current_map_, transformed_incoming, voxel_resolution_);
  } else if (policy == "CONFIDENCE_WEIGHTED") {
    result = mergeConfidenceWeighted(current_map_, transformed_incoming,
                                      voxel_resolution_, msg->icp_fitness_weight);
  } else {
    result = mergeTimestampPriority(current_map_, transformed_incoming, voxel_resolution_);
  }

  result.header.stamp = this->now();

  // Publish merged map
  if (result.success) {
    sensor_msgs::msg::PointCloud2 merged_msg;
    pcl::toROSMsg(*current_map_, merged_msg);
    merged_msg.header.stamp = this->now();
    merged_msg.header.frame_id = "map";
    pub_merged_map_->publish(merged_msg);
  }

  pub_completion_->publish(result);
}

void VoxelMergeNode::onRollback(const std_msgs::msg::Bool::SharedPtr msg)
{
  if (!msg->data) return;

  std::lock_guard<std::mutex> lock(mutex_);
  if (backup_map_) {
    current_map_ = std::make_shared<CloudType>(*backup_map_);
    RCLCPP_WARN(this->get_logger(), "Map rolled back to pre-merge state (%zu pts)",
                current_map_->size());

    sensor_msgs::msg::PointCloud2 restored_msg;
    pcl::toROSMsg(*current_map_, restored_msg);
    restored_msg.header.stamp = this->now();
    restored_msg.header.frame_id = "map";
    pub_merged_map_->publish(restored_msg);
  } else {
    RCLCPP_ERROR(this->get_logger(), "No backup map available for rollback");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// TIMESTAMP_PRIORITY: incoming (newer) wins in overlap regions
// ─────────────────────────────────────────────────────────────────────────────
fastlio2_health::msg::MergeCompletion VoxelMergeNode::mergeTimestampPriority(
  CloudType::Ptr existing, CloudType::Ptr incoming, float voxel_res)
{
  auto existing_voxels = buildVoxelMap(existing, voxel_res);
  auto incoming_voxels = buildVoxelMap(incoming, voxel_res);

  int points_added = 0;
  int points_removed = 0;
  int overlap_voxels = 0;
  int total_incoming_voxels = static_cast<int>(incoming_voxels.size());

  // Remove existing points in voxels that overlap with incoming
  for (const auto& [key, _] : incoming_voxels) {
    auto it = existing_voxels.find(key);
    if (it != existing_voxels.end()) {
      points_removed += static_cast<int>(it->second.size());
      existing_voxels.erase(it);
      overlap_voxels++;
    }
  }

  // Rebuild cloud: existing non-overlap + all incoming
  auto merged = std::make_shared<CloudType>();
  for (const auto& [_, pts] : existing_voxels) {
    for (const auto& pt : pts) merged->push_back(pt);
  }
  for (const auto& [_, pts] : incoming_voxels) {
    for (const auto& pt : pts) merged->push_back(pt);
    points_added += static_cast<int>(pts.size());
  }

  *existing = *merged;

  fastlio2_health::msg::MergeCompletion result;
  result.success = true;
  result.points_added = points_added;
  result.points_removed = points_removed;
  result.overlap_ratio = (total_incoming_voxels > 0) ?
    static_cast<float>(overlap_voxels) / total_incoming_voxels : 0.0f;

  RCLCPP_INFO(this->get_logger(),
    "TIMESTAMP_PRIORITY merge: +%d/-%d pts, overlap=%.2f",
    points_added, points_removed, result.overlap_ratio);

  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// DENSITY_PRIORITY: per-voxel, side with more points wins
// ─────────────────────────────────────────────────────────────────────────────
fastlio2_health::msg::MergeCompletion VoxelMergeNode::mergeDensityPriority(
  CloudType::Ptr existing, CloudType::Ptr incoming, float voxel_res)
{
  auto existing_voxels = buildVoxelMap(existing, voxel_res);
  auto incoming_voxels = buildVoxelMap(incoming, voxel_res);

  int points_added = 0;
  int points_removed = 0;
  int overlap_voxels = 0;
  int total_incoming_voxels = static_cast<int>(incoming_voxels.size());

  auto merged = std::make_shared<CloudType>();

  // Process all existing voxels
  for (auto& [key, existing_pts] : existing_voxels) {
    auto it = incoming_voxels.find(key);
    if (it != incoming_voxels.end()) {
      overlap_voxels++;
      // Density comparison: more points per voxel wins
      if (it->second.size() > existing_pts.size()) {
        // Incoming wins
        points_removed += static_cast<int>(existing_pts.size());
        points_added += static_cast<int>(it->second.size());
        for (const auto& pt : it->second) merged->push_back(pt);
      } else {
        // Existing wins
        for (const auto& pt : existing_pts) merged->push_back(pt);
      }
      incoming_voxels.erase(it);
    } else {
      // No overlap, keep existing
      for (const auto& pt : existing_pts) merged->push_back(pt);
    }
  }

  // Add remaining incoming voxels (no overlap)
  for (const auto& [_, pts] : incoming_voxels) {
    points_added += static_cast<int>(pts.size());
    for (const auto& pt : pts) merged->push_back(pt);
  }

  *existing = *merged;

  fastlio2_health::msg::MergeCompletion result;
  result.success = true;
  result.points_added = points_added;
  result.points_removed = points_removed;
  result.overlap_ratio = (total_incoming_voxels > 0) ?
    static_cast<float>(overlap_voxels) / total_incoming_voxels : 0.0f;

  RCLCPP_INFO(this->get_logger(),
    "DENSITY_PRIORITY merge: +%d/-%d pts, overlap=%.2f",
    points_added, points_removed, result.overlap_ratio);

  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// CONFIDENCE_WEIGHTED: blend by ICP fitness weight
// ─────────────────────────────────────────────────────────────────────────────
fastlio2_health::msg::MergeCompletion VoxelMergeNode::mergeConfidenceWeighted(
  CloudType::Ptr existing, CloudType::Ptr incoming,
  float voxel_res, float icp_fitness_weight)
{
  auto existing_voxels = buildVoxelMap(existing, voxel_res);
  auto incoming_voxels = buildVoxelMap(incoming, voxel_res);

  // Survival probability: lower fitness score = higher confidence = higher survival
  float incoming_survival = std::clamp(1.0f - icp_fitness_weight, 0.0f, 1.0f);

  int points_added = 0;
  int points_removed = 0;
  int overlap_voxels = 0;
  int total_incoming_voxels = static_cast<int>(incoming_voxels.size());

  std::mt19937 rng(42);  // deterministic for reproducibility
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  auto merged = std::make_shared<CloudType>();

  for (auto& [key, existing_pts] : existing_voxels) {
    auto it = incoming_voxels.find(key);
    if (it != incoming_voxels.end()) {
      overlap_voxels++;
      // Blend: probabilistically keep points from each side
      for (const auto& pt : existing_pts) {
        if (dist(rng) > incoming_survival) {
          merged->push_back(pt);
        } else {
          points_removed++;
        }
      }
      for (const auto& pt : it->second) {
        if (dist(rng) < incoming_survival) {
          merged->push_back(pt);
          points_added++;
        }
      }
      incoming_voxels.erase(it);
    } else {
      for (const auto& pt : existing_pts) merged->push_back(pt);
    }
  }

  // Non-overlapping incoming voxels
  for (const auto& [_, pts] : incoming_voxels) {
    points_added += static_cast<int>(pts.size());
    for (const auto& pt : pts) merged->push_back(pt);
  }

  *existing = *merged;

  fastlio2_health::msg::MergeCompletion result;
  result.success = true;
  result.points_added = points_added;
  result.points_removed = points_removed;
  result.overlap_ratio = (total_incoming_voxels > 0) ?
    static_cast<float>(overlap_voxels) / total_incoming_voxels : 0.0f;

  RCLCPP_INFO(this->get_logger(),
    "CONFIDENCE_WEIGHTED merge: +%d/-%d pts, overlap=%.2f, survival=%.2f",
    points_added, points_removed, result.overlap_ratio, incoming_survival);

  return result;
}

}  // namespace fastlio2_health

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fastlio2_health::VoxelMergeNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
