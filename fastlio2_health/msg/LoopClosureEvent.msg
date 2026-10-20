std_msgs/Header header
int32 node_i
int32 node_j
float32 match_score        # 0.0-1.0
float32 correction_magnitude_m
geometry_msgs/TransformStamped relative_transform
bool accepted              # whether the closure passed the acceptance threshold
