#include "icp_localizer.h"

ICPLocalizer::ICPLocalizer(const ICPConfig &config) : m_config(config)
{
    m_refine_inp.reset(new CloudType);
    m_refine_tgt.reset(new CloudType);
    m_rough_inp.reset(new CloudType);
    m_rough_tgt.reset(new CloudType);

    m_active_rough_corr_dist  = m_config.rough_max_corr_dist;
    m_active_refine_corr_dist = m_config.refine_max_corr_dist;
}
bool ICPLocalizer::loadMap(const std::string &path)
{
    if (!std::filesystem::exists(path))
    {
        std::cerr << "Map file not found: " << path << std::endl;
        return false;
    }
    pcl::PCDReader reader;
    CloudType::Ptr cloud(new CloudType);
    reader.read(path, *cloud);
    if (m_config.refine_map_resolution > 0)
    {
        m_voxel_filter.setLeafSize(m_config.refine_map_resolution, m_config.refine_map_resolution, m_config.refine_map_resolution);
        m_voxel_filter.setInputCloud(cloud);
        m_voxel_filter.filter(*m_refine_tgt);
    }
    else
    {
        pcl::copyPointCloud(*cloud, *m_refine_tgt);
    }

    if (m_config.rough_map_resolution > 0)
    {
        m_voxel_filter.setLeafSize(m_config.rough_map_resolution, m_config.rough_map_resolution, m_config.rough_map_resolution);
        m_voxel_filter.setInputCloud(cloud);
        m_voxel_filter.filter(*m_rough_tgt);
    }
    else
    {
        pcl::copyPointCloud(*cloud, *m_rough_tgt);
    }
    return true;
}
void ICPLocalizer::setInput(const CloudType::Ptr &cloud)
{
    if (m_config.refine_scan_resolution > 0)
    {
        m_voxel_filter.setLeafSize(m_config.refine_scan_resolution, m_config.refine_scan_resolution, m_config.refine_scan_resolution);
        m_voxel_filter.setInputCloud(cloud);
        m_voxel_filter.filter(*m_refine_inp);
    }
    else
    {
        pcl::copyPointCloud(*cloud, *m_refine_inp);
    }

    if (m_config.rough_scan_resolution > 0)
    {
        m_voxel_filter.setLeafSize(m_config.rough_scan_resolution, m_config.rough_scan_resolution, m_config.rough_scan_resolution);
        m_voxel_filter.setInputCloud(cloud);
        m_voxel_filter.filter(*m_rough_inp);
    }
    else
    {
        pcl::copyPointCloud(*cloud, *m_rough_inp);
    }
}

bool ICPLocalizer::align(M4F &guess)
{
    CloudType::Ptr aligned_cloud(new CloudType);
    if (m_refine_tgt->size() == 0 || m_rough_tgt->size() == 0)
        return false;

    // ── Rough stage ───────────────────────────────────────────────────────────
    m_rough_icp.setMaximumIterations(m_config.rough_max_iteration);
    m_rough_icp.setMaxCorrespondenceDistance(m_active_rough_corr_dist);
    m_rough_icp.setInputSource(m_rough_inp);
    m_rough_icp.setInputTarget(m_rough_tgt);
    m_rough_icp.align(*aligned_cloud, guess);

    if (!m_rough_icp.hasConverged() || m_rough_icp.getFitnessScore() > m_config.rough_score_thresh)
        return false;

    // If rough fitness is above the confidence threshold the initial guess is
    // considered marginal: widen the refine correspondence distance to avoid
    // diverging from a slightly-off starting point.
    double rough_score  = m_rough_icp.getFitnessScore();
    bool   marginal     = rough_score > m_config.rough_confidence_thresh;
    double refine_corr  = marginal
        ? m_active_refine_corr_dist * m_config.recovery_corr_scale
        : m_active_refine_corr_dist;

    // ── Refine stage ──────────────────────────────────────────────────────────
    m_refine_icp.setMaximumIterations(m_config.refine_max_iteration);
    m_refine_icp.setMaxCorrespondenceDistance(refine_corr);
    m_refine_icp.setInputSource(m_refine_inp);
    m_refine_icp.setInputTarget(m_refine_tgt);
    m_refine_icp.align(*aligned_cloud, m_rough_icp.getFinalTransformation());

    if (!m_refine_icp.hasConverged() || m_refine_icp.getFitnessScore() > m_config.refine_score_thresh)
        return false;

    m_last_fitness = m_refine_icp.getFitnessScore();
    guess = m_refine_icp.getFinalTransformation();
    return true;
}