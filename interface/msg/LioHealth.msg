# Health of one LiDAR-inertial update, published with every odometry sample
# (same stamp). Roadmap B1. A consumer should treat a missing message as bad.

std_msgs/Header header

# False when the scan gave the filter nothing to work with (no point matched a
# plane). The odometry for this stamp is then IMU prediction only.
bool valid

uint32 points                 # points offered to the filter (downsampled scan)
uint32 matched                # points matched to a map plane in the last iteration
float32 match_ratio           # matched / points, 0 if points is 0
float32 residual_rms          # RMS point-to-plane distance of the matched points (m)
uint8 iterations              # filter iterations used

# How well the matched planes constrain motion, from the measurement part of
# the filter's information matrix. Each value is 3 * smallest eigenvalue /
# trace of its 3x3 block: 1.0 when all directions are equally constrained,
# 0.0 when one direction is not constrained at all.
float32 translation_degeneracy
float32 rotation_degeneracy
# World-frame unit vector of the least constrained translation direction
# (for a plain corridor: along the corridor).
geometry_msgs/Vector3 weakest_translation_dir
