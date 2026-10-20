#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>

#include "fastlio2_health/msg/merge_trigger.hpp"
#include "fastlio2_health/msg/merge_completion.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

#include <mutex>
#include <string>

namespace fastlio2_health
{

using PointType = pcl::PointXYZI;
using CloudType = pcl::PointCloud<PointType>;

class VoxelMergeNode : public rclcpp::Node
{
public:
  explicit VoxelMergeNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
  void onMergeTrigger(const fastlio2_health::msg::MergeTrigger::SharedPtr msg);
  void onRollback(const std_msgs::msg::Bool::SharedPtr msg);
  void onCurrentMap(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  // Merge strategies
  fastlio2_health::msg::MergeCompletion mergeTimestampPriority(
    CloudType::Ptr existing, CloudType::Ptr incoming, float voxel_res);
  fastlio2_health::msg::MergeCompletion mergeDensityPriority(
    CloudType::Ptr existing, CloudType::Ptr incoming, float voxel_res);
  fastlio2_health::msg::MergeCompletion mergeConfidenceWeighted(
    CloudType::Ptr existing, CloudType::Ptr incoming,
    float voxel_res, float icp_fitness_weight);

  // Parameters
  std::string merge_policy_;
  float voxel_resolution_;

  // State
  std::mutex mutex_;
  CloudType::Ptr current_map_;
  CloudType::Ptr backup_map_;  // for rollback

  // ROS interfaces
  rclcpp::Subscription<fastlio2_health::msg::MergeTrigger>::SharedPtr sub_trigger_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_rollback_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_current_map_;
  rclcpp::Publisher<fastlio2_health::msg::MergeCompletion>::SharedPtr pub_completion_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_merged_map_;
};

}  // namespace fastlio2_health
