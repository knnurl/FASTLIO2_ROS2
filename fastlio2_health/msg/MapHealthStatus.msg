std_msgs/Header header
float32 global_health_score          # 0.0 (degraded) to 1.0 (healthy)
int32 num_degraded_cells
int32 num_cells_pending_hba
int32 num_cells_pending_merge
string current_phase                 # MONITORING | HBA_RUNNING | MERGING | VERIFYING | ROLLBACK
geometry_msgs/Point worst_cell_centre
float32 worst_cell_score
