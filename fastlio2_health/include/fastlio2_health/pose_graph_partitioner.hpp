#pragma once

#include <Eigen/Core>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
#include <cmath>

namespace fastlio2_health
{

class PoseGraphPartitioner
{
public:
  explicit PoseGraphPartitioner(double cell_size_m = 50.0);

  void insertNode(int node_id, const Eigen::Vector3d& position);

  std::vector<int> getSubgraphNodes(const Eigen::Vector3d& loop_closure_position,
                                     int radius_cells = 1) const;

  void removeNode(int node_id);

  size_t totalNodes() const;
  size_t totalCells() const;

private:
  struct CellKey
  {
    int cx, cy;

    bool operator==(const CellKey& other) const
    {
      return cx == other.cx && cy == other.cy;
    }
  };

  struct CellKeyHash
  {
    size_t operator()(const CellKey& k) const
    {
      auto h1 = std::hash<int>{}(k.cx);
      auto h2 = std::hash<int>{}(k.cy);
      return h1 ^ (h2 << 16);
    }
  };

  CellKey positionToCell(const Eigen::Vector3d& pos) const;

  double cell_size_m_;
  mutable std::mutex mutex_;

  // cell -> set of node IDs
  std::unordered_map<CellKey, std::unordered_set<int>, CellKeyHash> cell_to_nodes_;
  // node_id -> cell key (for removal)
  std::unordered_map<int, CellKey> node_to_cell_;
};

}  // namespace fastlio2_health
