#include "fastlio2_health/map_staleness_tracker.hpp"
#include <algorithm>
#include <limits>

namespace fastlio2_health
{

MapStalenessTracker::MapStalenessTracker(double cell_size_m,
                                          double staleness_decay_s,
                                          double confidence_threshold,
                                          double staleness_hba_threshold)
  : cell_size_m_(cell_size_m)
  , staleness_decay_s_(staleness_decay_s)
  , confidence_threshold_(confidence_threshold)
  , staleness_hba_threshold_(staleness_hba_threshold)
{
}

CellCoord MapStalenessTracker::toCell(double x, double y) const
{
  return {static_cast<int>(std::floor(x / cell_size_m_)),
          static_cast<int>(std::floor(y / cell_size_m_))};
}

void MapStalenessTracker::updateConfidentLocalisation(
  double x, double y, const rclcpp::Time& stamp, const std::string& session_id)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto cell = toCell(x, y);
  auto& region = regions_[cell];
  region.last_confident_localisation = stamp;
  region.session_id = session_id;
}

void MapStalenessTracker::updateStaleness(const rclcpp::Time& now)
{
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [coord, region] : regions_) {
    if (region.last_confident_localisation.nanoseconds() == 0) {
      region.staleness_score = 1.0f;
      continue;
    }
    double elapsed = (now - region.last_confident_localisation).seconds();
    region.staleness_score = static_cast<float>(
      std::min(1.0, elapsed / staleness_decay_s_));
    region.flagged_for_hba = (region.staleness_score > staleness_hba_threshold_);
  }
}

std::vector<CellCoord> MapStalenessTracker::getStaleCells() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<CellCoord> result;
  for (const auto& [coord, region] : regions_) {
    if (region.flagged_for_hba) {
      result.push_back(coord);
    }
  }
  return result;
}

float MapStalenessTracker::getStalenessScore(int cx, int cy) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  CellCoord key{cx, cy};
  auto it = regions_.find(key);
  if (it == regions_.end()) return 1.0f;
  return it->second.staleness_score;
}

const std::unordered_map<CellCoord, MapRegionMetadata, CellCoordHash>&
MapStalenessTracker::getRegions() const
{
  return regions_;
}

void MapStalenessTracker::setParams(double staleness_decay_s,
                                     double confidence_threshold,
                                     double staleness_hba_threshold)
{
  std::lock_guard<std::mutex> lock(mutex_);
  staleness_decay_s_ = staleness_decay_s;
  confidence_threshold_ = confidence_threshold;
  staleness_hba_threshold_ = staleness_hba_threshold;
}

std::vector<int8_t> MapStalenessTracker::toOccupancyData(
  int& width, int& height, double& origin_x, double& origin_y) const
{
  std::lock_guard<std::mutex> lock(mutex_);

  if (regions_.empty()) {
    width = height = 1;
    origin_x = origin_y = 0.0;
    return {0};
  }

  int min_cx = std::numeric_limits<int>::max();
  int max_cx = std::numeric_limits<int>::min();
  int min_cy = std::numeric_limits<int>::max();
  int max_cy = std::numeric_limits<int>::min();

  for (const auto& [coord, _] : regions_) {
    min_cx = std::min(min_cx, coord.x);
    max_cx = std::max(max_cx, coord.x);
    min_cy = std::min(min_cy, coord.y);
    max_cy = std::max(max_cy, coord.y);
  }

  width = max_cx - min_cx + 1;
  height = max_cy - min_cy + 1;
  origin_x = min_cx * cell_size_m_;
  origin_y = min_cy * cell_size_m_;

  std::vector<int8_t> data(width * height, -1);  // unknown = -1
  for (const auto& [coord, region] : regions_) {
    int idx = (coord.y - min_cy) * width + (coord.x - min_cx);
    data[idx] = static_cast<int8_t>(region.staleness_score * 100.0f);
  }
  return data;
}

}  // namespace fastlio2_health
