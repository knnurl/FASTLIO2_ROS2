#pragma once
#include "commons.h"
#include <future>
#include <optional>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/common/common.h>

struct GlobalLocConfig
{
    // Coarse grid search
    double xy_step_m       = 2.0;   // XY grid step in meters
    double yaw_step_deg    = 30.0;  // Yaw step in degrees
    double z_offset_m      = 0.0;   // Z offset relative to map centroid

    // NDT coarse scoring
    double ndt_resolution  = 1.0;   // NDT voxel resolution
    int    ndt_max_iter    = 15;    // NDT iters to refine after pre-transform
    double ndt_score_thresh = 10.0; // Accept coarse if score < this
    int    top_k           = 5;     // Top-K candidates to refine with ICP

    // Prior pose fast check (used on re-launch with saved pose)
    double prior_icp_max_corr_dist = 3.0;  // Wider window for prior-seeded ICP
    double prior_icp_fitness_thresh = 0.4; // Accept prior if fitness < this

    // ICP refinement
    double icp_max_corr_dist = 1.5;
    int    icp_max_iter      = 100;
    double icp_fitness_thresh = 0.5; // Accept final if score < this

    // Input cloud downsampling
    double scan_voxel_coarse = 0.4; // For NDT scoring
    double scan_voxel_refine = 0.15;// For ICP refinement
    double map_voxel         = 0.25;// For map loading
};

struct PoseCandidate
{
    Eigen::Matrix4f transform;
    double ndt_score;
    double icp_fitness;
};

class GlobalLocalizer
{
public:
    explicit GlobalLocalizer(const GlobalLocConfig &cfg);

    // Load map from PCD file; returns false on error.
    bool loadMap(const std::string &path);

    // Run global localization on the given body-frame scan.
    // body_rot: current rotation of the body in the LIO world frame (from odometry).
    //   The map and LIO world frames are both gravity-aligned, so their relative
    //   transform is a pure yaw + translation.  Roll and pitch from body_rot are
    //   therefore the same in the map frame and are baked into every search
    //   candidate, so only yaw is swept.  Pass Identity() to fall back to the old
    //   horizontal-only behaviour (not recommended when the LiDAR is tilted).
    std::optional<Eigen::Matrix4f> localize(
        CloudType::Ptr scan,
        const Eigen::Matrix3f &body_rot = Eigen::Matrix3f::Identity());

    // Quick ICP from a prior pose guess (e.g. saved from last session).
    // Returns refined transform or nullopt if ICP fitness exceeds threshold.
    std::optional<Eigen::Matrix4f> localizeFromPrior(CloudType::Ptr scan,
                                                     const Eigen::Matrix4f &prior);

    bool isMapLoaded() const { return m_map_loaded; }
    CloudType::Ptr map() const { return m_map_cloud; }

    // Fitness score from the most recent localize() or localizeFromPrior() call.
    double getLastFitness() const { return m_last_fitness; }

private:
    std::vector<PoseCandidate> generateCandidates(const Eigen::Vector3f &centroid,
                                                   const Eigen::Matrix3f &body_rot);

    // Pre-transform scan to candidate pose, run NDT, return {score, refined_transform}.
    // Scoring happens AT the candidate position, and NDT refines it slightly.
    std::pair<double, Eigen::Matrix4f> scoreNDT(CloudType::Ptr scan_coarse,
                                                const Eigen::Matrix4f &candidate);

    // Refine a candidate with full ICP.
    PoseCandidate refineICP(CloudType::Ptr scan_refine, const Eigen::Matrix4f &guess);

    GlobalLocConfig m_cfg;
    bool m_map_loaded = false;
    double m_last_fitness = 0.0;

    CloudType::Ptr m_map_cloud;      // Full-res map (for ICP)
    CloudType::Ptr m_map_coarse;     // Downsampled for NDT

    // NDT object (not thread-safe; used in scoring loop single-threaded)
    pcl::NormalDistributionsTransform<PointType, PointType> m_ndt;
    pcl::VoxelGrid<PointType> m_voxel;
};
