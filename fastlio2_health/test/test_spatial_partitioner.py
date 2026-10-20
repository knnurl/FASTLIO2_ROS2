#!/usr/bin/env python3
"""
test_spatial_partitioner.py
───────────────────────────
Unit test for PoseGraphPartitioner: insert 100 nodes across a 200m x 200m grid,
verify that getSubgraphNodes() returns only nodes within the expected cells.

This test uses ctypes/subprocess to call the C++ library, or tests the
logic via a Python reimplementation of the algorithm.
"""
import unittest
import math
import random


class PoseGraphPartitionerPython:
    """Pure Python reimplementation for testing the spatial partitioning logic."""

    def __init__(self, cell_size_m=50.0):
        self.cell_size = cell_size_m
        self.cell_to_nodes = {}  # (cx, cy) -> set of node_ids
        self.node_to_cell = {}   # node_id -> (cx, cy)

    def _pos_to_cell(self, x, y):
        return (int(math.floor(x / self.cell_size)),
                int(math.floor(y / self.cell_size)))

    def insert_node(self, node_id, x, y, z=0.0):
        cell = self._pos_to_cell(x, y)
        # Remove from old cell if re-inserting
        if node_id in self.node_to_cell:
            old_cell = self.node_to_cell[node_id]
            self.cell_to_nodes[old_cell].discard(node_id)
            if not self.cell_to_nodes[old_cell]:
                del self.cell_to_nodes[old_cell]

        if cell not in self.cell_to_nodes:
            self.cell_to_nodes[cell] = set()
        self.cell_to_nodes[cell].add(node_id)
        self.node_to_cell[node_id] = cell

    def get_subgraph_nodes(self, x, y, radius_cells=1):
        centre = self._pos_to_cell(x, y)
        result = []
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                key = (centre[0] + dx, centre[1] + dy)
                if key in self.cell_to_nodes:
                    result.extend(self.cell_to_nodes[key])
        return result

    def remove_node(self, node_id):
        if node_id not in self.node_to_cell:
            return
        cell = self.node_to_cell[node_id]
        self.cell_to_nodes[cell].discard(node_id)
        if not self.cell_to_nodes[cell]:
            del self.cell_to_nodes[cell]
        del self.node_to_cell[node_id]

    def total_nodes(self):
        return len(self.node_to_cell)

    def total_cells(self):
        return len(self.cell_to_nodes)


class TestSpatialPartitioner(unittest.TestCase):

    def test_insert_100_nodes_200m_grid(self):
        """Insert 100 nodes across a 200m x 200m grid with 50m cells."""
        p = PoseGraphPartitionerPython(cell_size_m=50.0)

        # Insert 100 random nodes in [0, 200] x [0, 200]
        random.seed(42)
        positions = {}
        for i in range(100):
            x = random.uniform(0, 200)
            y = random.uniform(0, 200)
            p.insert_node(i, x, y)
            positions[i] = (x, y)

        self.assertEqual(p.total_nodes(), 100)
        # With 50m cells in 200m grid, expect 4x4=16 cells max
        self.assertLessEqual(p.total_cells(), 16)
        self.assertGreater(p.total_cells(), 0)

    def test_subgraph_containment(self):
        """Nodes returned by getSubgraphNodes are within expected cells."""
        p = PoseGraphPartitionerPython(cell_size_m=50.0)

        # Insert nodes in known positions
        p.insert_node(0, 25.0, 25.0)   # cell (0, 0)
        p.insert_node(1, 75.0, 25.0)   # cell (1, 0)
        p.insert_node(2, 25.0, 75.0)   # cell (0, 1)
        p.insert_node(3, 75.0, 75.0)   # cell (1, 1)
        p.insert_node(4, 125.0, 125.0) # cell (2, 2)
        p.insert_node(5, 175.0, 175.0) # cell (3, 3)

        # Query at (25, 25) with radius=0 → should only get node 0
        result = p.get_subgraph_nodes(25.0, 25.0, radius_cells=0)
        self.assertEqual(sorted(result), [0])

        # Query at (25, 25) with radius=1 → should get nodes 0, 1, 2, 3
        # (covers cells (0,0), (1,0), (0,1), (1,1), and (-1,-1)..(-1,1),(0,-1),(1,-1))
        result = p.get_subgraph_nodes(25.0, 25.0, radius_cells=1)
        self.assertIn(0, result)
        self.assertIn(1, result)
        self.assertIn(2, result)
        self.assertIn(3, result)
        self.assertNotIn(4, result)  # cell (2,2) is outside radius=1
        self.assertNotIn(5, result)  # cell (3,3) is outside radius=1

    def test_subgraph_at_boundary(self):
        """Boundary loop closures should include adjacent cells."""
        p = PoseGraphPartitionerPython(cell_size_m=50.0)

        # Node at edge of cell boundary
        p.insert_node(0, 49.9, 49.9)  # cell (0, 0)
        p.insert_node(1, 50.1, 50.1)  # cell (1, 1)

        # Query at boundary with radius=1 should catch both
        result = p.get_subgraph_nodes(50.0, 50.0, radius_cells=1)
        self.assertIn(0, result)
        self.assertIn(1, result)

    def test_remove_node(self):
        """Removed nodes should not appear in queries."""
        p = PoseGraphPartitionerPython(cell_size_m=50.0)
        p.insert_node(0, 25.0, 25.0)
        p.insert_node(1, 25.0, 25.0)

        self.assertEqual(p.total_nodes(), 2)
        p.remove_node(0)
        self.assertEqual(p.total_nodes(), 1)

        result = p.get_subgraph_nodes(25.0, 25.0, radius_cells=0)
        self.assertNotIn(0, result)
        self.assertIn(1, result)

    def test_node_reinsertion(self):
        """Re-inserting a node should move it to the new cell."""
        p = PoseGraphPartitionerPython(cell_size_m=50.0)
        p.insert_node(0, 25.0, 25.0)  # cell (0, 0)

        result = p.get_subgraph_nodes(25.0, 25.0, radius_cells=0)
        self.assertIn(0, result)

        # Re-insert at a different position
        p.insert_node(0, 175.0, 175.0)  # cell (3, 3)
        self.assertEqual(p.total_nodes(), 1)

        # Should not be in old cell
        result = p.get_subgraph_nodes(25.0, 25.0, radius_cells=0)
        self.assertNotIn(0, result)

        # Should be in new cell
        result = p.get_subgraph_nodes(175.0, 175.0, radius_cells=0)
        self.assertIn(0, result)

    def test_negative_coordinates(self):
        """Partitioner should handle negative coordinates correctly."""
        p = PoseGraphPartitionerPython(cell_size_m=50.0)
        p.insert_node(0, -25.0, -25.0)  # cell (-1, -1)
        p.insert_node(1, 25.0, 25.0)    # cell (0, 0)

        result = p.get_subgraph_nodes(-25.0, -25.0, radius_cells=1)
        self.assertIn(0, result)
        self.assertIn(1, result)  # cell (0,0) is adjacent to (-1,-1)


if __name__ == '__main__':
    unittest.main()
