#include "fastlio2_health/pose_graph_partitioner.hpp"

namespace fastlio2_health
{

PoseGraphPartitioner::PoseGraphPartitioner(double cell_size_m)
  : cell_size_m_(cell_size_m)
{
}

PoseGraphPartitioner::CellKey PoseGraphPartitioner::positionToCell(
  const Eigen::Vector3d& pos) const
{
  return {static_cast<int>(std::floor(pos.x() / cell_size_m_)),
          static_cast<int>(std::floor(pos.y() / cell_size_m_))};
}

void PoseGraphPartitioner::insertNode(int node_id, const Eigen::Vector3d& position)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto cell = positionToCell(position);

  // Remove from old cell if re-inserting
  auto it = node_to_cell_.find(node_id);
  if (it != node_to_cell_.end()) {
    cell_to_nodes_[it->second].erase(node_id);
    if (cell_to_nodes_[it->second].empty()) {
      cell_to_nodes_.erase(it->second);
    }
  }

  cell_to_nodes_[cell].insert(node_id);
  node_to_cell_[node_id] = cell;
}

std::vector<int> PoseGraphPartitioner::getSubgraphNodes(
  const Eigen::Vector3d& loop_closure_position, int radius_cells) const
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto centre = positionToCell(loop_closure_position);
  std::vector<int> result;

  for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
    for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
      CellKey key{centre.cx + dx, centre.cy + dy};
      auto it = cell_to_nodes_.find(key);
      if (it != cell_to_nodes_.end()) {
        result.insert(result.end(), it->second.begin(), it->second.end());
      }
    }
  }

  return result;
}

void PoseGraphPartitioner::removeNode(int node_id)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = node_to_cell_.find(node_id);
  if (it == node_to_cell_.end()) return;

  cell_to_nodes_[it->second].erase(node_id);
  if (cell_to_nodes_[it->second].empty()) {
    cell_to_nodes_.erase(it->second);
  }
  node_to_cell_.erase(it);
}

size_t PoseGraphPartitioner::totalNodes() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return node_to_cell_.size();
}

size_t PoseGraphPartitioner::totalCells() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return cell_to_nodes_.size();
}

}  // namespace fastlio2_health
