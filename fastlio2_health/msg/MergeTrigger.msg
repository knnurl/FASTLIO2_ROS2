std_msgs/Header header
sensor_msgs/PointCloud2 incoming_pcd
geometry_msgs/TransformStamped transform
string merge_policy   # TIMESTAMP_PRIORITY | DENSITY_PRIORITY | CONFIDENCE_WEIGHTED
float32 icp_fitness_weight   # used only for CONFIDENCE_WEIGHTED
