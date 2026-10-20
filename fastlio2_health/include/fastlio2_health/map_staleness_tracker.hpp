#pragma once

#include <rclcpp/rclcpp.hpp>
#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
#include <cmath>

namespace fastlio2_health
{

struct MapRegionMetadata
{
  rclcpp::Time last_confident_localisation{0, 0, RCL_ROS_TIME};
  std::string session_id;
  float staleness_score = 1.0f;
  bool flagged_for_hba = false;
};

struct CellCoord
{
  int x, y;
  bool operator==(const CellCoord& o) const { return x == o.x && y == o.y; }
};

struct CellCoordHash
{
  size_t operator()(const CellCoord& c) const
  {
    return std::hash<int>{}(c.x) ^ (std::hash<int>{}(c.y) << 16);
  }
};

class MapStalenessTracker
{
public:
  explicit MapStalenessTracker(double cell_size_m = 10.0,
                                double staleness_decay_s = 300.0,
                                double confidence_threshold = 0.7,
                                double staleness_hba_threshold = 0.8);

  void updateConfidentLocalisation(double x, double y, const rclcpp::Time& stamp,
                                    const std::string& session_id = "default");

  void updateStaleness(const rclcpp::Time& now);

  std::vector<CellCoord> getStaleCells() const;

  float getStalenessScore(int cx, int cy) const;

  const std::unordered_map<CellCoord, MapRegionMetadata, CellCoordHash>& getRegions() const;

  void setParams(double staleness_decay_s, double confidence_threshold,
                 double staleness_hba_threshold);

  std::vector<int8_t> toOccupancyData(int& width, int& height,
                                       double& origin_x, double& origin_y) const;

  double cellSize() const { return cell_size_m_; }

private:
  CellCoord toCell(double x, double y) const;

  double cell_size_m_;
  double staleness_decay_s_;
  double confidence_threshold_;
  double staleness_hba_threshold_;
  mutable std::mutex mutex_;

  std::unordered_map<CellCoord, MapRegionMetadata, CellCoordHash> regions_;
};

}  // namespace fastlio2_health
