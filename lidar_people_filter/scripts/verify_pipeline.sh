#!/bin/bash
# Pre-flight checks for the people filter pipeline.
# Run this AFTER FastLIO is already publishing, BEFORE launching the filter.
# Usage: bash scripts/verify_pipeline.sh

set -e
echo "=== LiDAR people filter — pre-flight verification ==="

echo ""
echo "--- 1. Checking /fastlio2/world_cloud is publishing ---"
timeout 5 ros2 topic hz /fastlio2/world_cloud --once 2>/dev/null && echo "OK" || echo "FAIL — FastLIO not running?"

echo ""
echo "--- 2. Checking frame_id of /fastlio2/world_cloud ---"
FRAME=$(timeout 5 ros2 topic echo --once /fastlio2/world_cloud 2>/dev/null | grep "frame_id" | awk '{print $2}' | tr -d "'")
echo "frame_id = '${FRAME}'"
echo "  -> Set this in config/octomap.yaml as frame_id, and pass as world_frame launch arg"

echo ""
echo "--- 3. Checking QoS of /fastlio2/world_cloud ---"
timeout 5 ros2 topic info /fastlio2/world_cloud --verbose 2>/dev/null | grep -A2 "Publisher" | grep "Reliability" || echo "Could not check QoS"
echo "  -> If RELIABLE, change cloud_in_qos_reliability to 'reliable' in config/octomap.yaml"

echo ""
echo "--- 4. Checking TF chain ---"
timeout 5 ros2 run tf2_tools view_frames 2>/dev/null && echo "frames.pdf written to current directory" || echo "Could not generate TF frames"

echo ""
echo "--- 5. Checking octomap_server is installed ---"
ros2 pkg list | grep octomap_server && echo "OK" || echo "FAIL — run: sudo apt install ros-humble-octomap-server"

echo ""
echo "--- 6. Checking scikit-learn is installed ---"
python3 -c "import sklearn; print(f'scikit-learn {sklearn.__version__} OK')" || echo "FAIL — run: pip3 install scikit-learn"

echo ""
echo "=== Pre-flight complete. Fix any FAIL items before launching. ==="
