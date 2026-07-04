#pragma once
#include "commons.h"
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>

struct ICPConfig
{
    double refine_scan_resolution = 0.1;
    double refine_map_resolution = 0.1;
    double refine_score_thresh = 0.3;
    int refine_max_iteration = 50;
    double refine_max_corr_dist = 1.5;

    double rough_scan_resolution = 0.25;
    double rough_map_resolution = 0.25;
    double rough_score_thresh = 0.5;
    int rough_max_iteration = 30;
    double rough_max_corr_dist = 3.0;

    // If rough ICP fitness is above this (but still below rough_score_thresh), the
    // initial guess is considered marginal.  Refine stage then uses a wider
    // correspondence distance (refine_max_corr_dist * refine_marginal_scale) to
    // avoid diverging from a slightly-off rough result.
    // Set to rough_score_thresh to disable (no marginal path).
    double rough_confidence_thresh = 0.25; // ~half of default rough_score_thresh

    // Multiplier applied to both correspondence distances when the tracking
    // recovery path is active (consecutive failures >= half of max_failures).
    double recovery_corr_scale = 1.5;
};

class ICPLocalizer
{
public:
    ICPLocalizer(const ICPConfig &config);
    
    bool loadMap(const std::string &path);
    
    void setInput(const CloudType::Ptr &cloud);

    bool align(M4F &guess);
    ICPConfig &config() { return m_config; }
    CloudType::Ptr roughMap() { return m_rough_tgt; }
    CloudType::Ptr refineMap() { return m_refine_tgt; }

    // Override correspondence distances at runtime (used for recovery widening).
    // Pass the original configured values to restore normal behaviour.
    void setCorrespondenceDists(double rough, double refine)
    {
        m_active_rough_corr_dist  = rough;
        m_active_refine_corr_dist = refine;
    }

    double getLastFitness() const { return m_last_fitness; }


private:
    ICPConfig m_config;

    // Active correspondence distances — may differ from config when recovery
    // widening is in effect.  Initialised from config in the constructor.
    double m_active_rough_corr_dist  = 3.0;
    double m_active_refine_corr_dist = 1.5;

    double m_last_fitness = 0.0;

    pcl::VoxelGrid<PointType> m_voxel_filter;
    pcl::IterativeClosestPoint<PointType, PointType> m_refine_icp;
    pcl::IterativeClosestPoint<PointType, PointType> m_rough_icp;
    CloudType::Ptr m_refine_inp;
    CloudType::Ptr m_rough_inp;
    CloudType::Ptr m_refine_tgt;
    CloudType::Ptr m_rough_tgt;
    std::string m_pcd_path;
};