#include "global_localizer.h"
#include <pcl/common/transforms.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <unordered_set>

GlobalLocalizer::GlobalLocalizer(const GlobalLocConfig &cfg) : m_cfg(cfg)
{
    m_map_cloud.reset(new CloudType);
    m_map_coarse.reset(new CloudType);
}

bool GlobalLocalizer::loadMap(const std::string &path)
{
    CloudType::Ptr raw(new CloudType);
    if (pcl::io::loadPCDFile<PointType>(path, *raw) < 0)
    {
        std::cerr << "[GlobalLocalizer] Failed to load: " << path << std::endl;
        return false;
    }

    if (m_cfg.map_voxel > 0)
    {
        m_voxel.setLeafSize(m_cfg.map_voxel, m_cfg.map_voxel, m_cfg.map_voxel);
        m_voxel.setInputCloud(raw);
        m_voxel.filter(*m_map_cloud);
    }
    else
    {
        m_map_cloud = raw;
    }

    // Coarser version for NDT scoring
    double coarse_leaf = std::max(m_cfg.map_voxel, m_cfg.ndt_resolution * 0.5);
    m_voxel.setLeafSize(coarse_leaf, coarse_leaf, coarse_leaf);
    m_voxel.setInputCloud(raw);
    m_voxel.filter(*m_map_coarse);

    // Build NDT index once (target never changes)
    m_ndt.setResolution(m_cfg.ndt_resolution);
    m_ndt.setMaximumIterations(m_cfg.ndt_max_iter);
    m_ndt.setStepSize(0.1);
    m_ndt.setTransformationEpsilon(0.01);
    m_ndt.setInputTarget(m_map_coarse);

    m_map_loaded = true;
    std::cout << "[GlobalLocalizer] Map loaded: " << m_map_cloud->size()
              << " pts (refined), " << m_map_coarse->size() << " pts (coarse)" << std::endl;
    return true;
}

std::vector<PoseCandidate> GlobalLocalizer::generateCandidates(const Eigen::Vector3f &map_centroid,
                                                                const Eigen::Matrix3f &body_rot)
{
    PointType min_pt, max_pt;
    pcl::getMinMax3D(*m_map_coarse, min_pt, max_pt);

    // Build a 2-D occupied-cell set from the coarse map so we can skip grid
    // positions that have no map content nearby.  Each map point votes for its
    // XY cell; a candidate position is kept only if its cell (or an adjacent
    // cell) is occupied.  This avoids scoring thousands of NDT candidates that
    // sit in empty corridors or outside the mapped area.
    auto cellKey = [](int ix, int iy) -> int64_t {
        return (static_cast<int64_t>(ix) << 32) | static_cast<uint32_t>(iy);
    };
    std::unordered_set<int64_t> occupied_cells;
    occupied_cells.reserve(m_map_coarse->size());
    for (const auto &pt : *m_map_coarse)
    {
        int ix = static_cast<int>(std::floor(pt.x / m_cfg.xy_step_m));
        int iy = static_cast<int>(std::floor(pt.y / m_cfg.xy_step_m));
        occupied_cells.insert(cellKey(ix, iy));
    }

    std::vector<PoseCandidate> candidates;
    double yaw_step_rad = m_cfg.yaw_step_deg * M_PI / 180.0;
    int skipped = 0;

    for (double x = min_pt.x; x <= max_pt.x; x += m_cfg.xy_step_m)
    {
        for (double y = min_pt.y; y <= max_pt.y; y += m_cfg.xy_step_m)
        {
            // Skip positions where neither this cell nor its 8 neighbours
            // contain any map points — nothing to align to here.
            int ix = static_cast<int>(std::floor(x / m_cfg.xy_step_m));
            int iy = static_cast<int>(std::floor(y / m_cfg.xy_step_m));
            bool has_content = false;
            for (int dx = -1; dx <= 1 && !has_content; ++dx)
                for (int dy = -1; dy <= 1 && !has_content; ++dy)
                    if (occupied_cells.count(cellKey(ix + dx, iy + dy)))
                        has_content = true;
            if (!has_content) { ++skipped; continue; }

            for (double yaw = 0.0; yaw < 2.0 * M_PI; yaw += yaw_step_rad)
            {
                // R_candidate = Rz(yaw) * body_rot
                //
                // The map and LIO world frames are both gravity-aligned, so their
                // relative rotation is a pure yaw.  Roll and pitch from body_rot
                // (measured by the IMU) are identical in both frames and must be
                // preserved in every candidate — only yaw is unknown.
                float cy = std::cos(yaw), sy = std::sin(yaw);
                Eigen::Matrix3f Rz;
                Rz << cy, -sy, 0.f,
                      sy,  cy, 0.f,
                       0.f, 0.f, 1.f;

                Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
                T.block<3, 3>(0, 0) = Rz * body_rot;
                T(0, 3) = static_cast<float>(x);
                T(1, 3) = static_cast<float>(y);
                T(2, 3) = map_centroid.z() + static_cast<float>(m_cfg.z_offset_m);
                candidates.push_back({T, 1e9, 1e9});
            }
        }
    }
    if (skipped > 0)
        std::cout << "[GlobalLocalizer] Occupied-mask: skipped " << skipped
                  << " empty XY cells, kept " << (candidates.size() / std::max(1, static_cast<int>(2*M_PI/yaw_step_rad)))
                  << " XY positions." << std::endl;
    return candidates;
}

// ─── Fixed NDT scoring ────────────────────────────────────────────────────────
// Pre-transform the scan to the candidate pose so NDT scores FROM that position,
// not from wherever a 1-iteration update happens to land.
// Returns {fitness_score, refined_transform}.
std::pair<double, Eigen::Matrix4f>
GlobalLocalizer::scoreNDT(CloudType::Ptr scan_coarse, const Eigen::Matrix4f &candidate)
{
    // Move scan to candidate position
    CloudType::Ptr transformed(new CloudType);
    pcl::transformPointCloud(*scan_coarse, *transformed, candidate);

    // NDT refines from there (identity initial guess since scan is already placed)
    CloudType aligned;
    m_ndt.setInputSource(transformed);
    m_ndt.align(aligned, Eigen::Matrix4f::Identity());

    if (!m_ndt.hasConverged())
        return {1e9, candidate};

    // Compose: ndt correction applied AFTER the candidate placement
    // T_final * p_body = T_ndt * (candidate * p_body)
    Eigen::Matrix4f refined = m_ndt.getFinalTransformation() * candidate;
    return {m_ndt.getFitnessScore(), refined};
}

PoseCandidate GlobalLocalizer::refineICP(CloudType::Ptr scan_refine, const Eigen::Matrix4f &guess)
{
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaximumIterations(m_cfg.icp_max_iter);
    icp.setMaxCorrespondenceDistance(m_cfg.icp_max_corr_dist);
    icp.setInputSource(scan_refine);
    icp.setInputTarget(m_map_cloud);

    CloudType aligned;
    icp.align(aligned, guess);

    PoseCandidate result;
    result.ndt_score = 0.0;
    if (icp.hasConverged())
    {
        result.transform   = icp.getFinalTransformation();
        result.icp_fitness = icp.getFitnessScore();
    }
    else
    {
        result.transform   = guess;
        result.icp_fitness = 1e9;
    }
    return result;
}

// ─── Prior-seeded ICP (for saved-pose warm start) ────────────────────────────
std::optional<Eigen::Matrix4f>
GlobalLocalizer::localizeFromPrior(CloudType::Ptr scan, const Eigen::Matrix4f &prior)
{
    if (!m_map_loaded || m_map_cloud->empty())
        return std::nullopt;

    CloudType::Ptr scan_ds(new CloudType);
    if (m_cfg.scan_voxel_refine > 0)
    {
        m_voxel.setLeafSize(m_cfg.scan_voxel_refine, m_cfg.scan_voxel_refine, m_cfg.scan_voxel_refine);
        m_voxel.setInputCloud(scan);
        m_voxel.filter(*scan_ds);
    }
    else
    {
        scan_ds = scan;
    }

    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaximumIterations(m_cfg.icp_max_iter);
    icp.setMaxCorrespondenceDistance(m_cfg.prior_icp_max_corr_dist);
    icp.setInputSource(scan_ds);
    icp.setInputTarget(m_map_cloud);

    CloudType aligned;
    icp.align(aligned, prior);

    if (!icp.hasConverged() || icp.getFitnessScore() > m_cfg.prior_icp_fitness_thresh)
    {
        std::cout << "[GlobalLocalizer] Prior ICP failed. fitness="
                  << icp.getFitnessScore() << " > thresh " << m_cfg.prior_icp_fitness_thresh
                  << ". Falling back to global search." << std::endl;
        return std::nullopt;
    }

    m_last_fitness = icp.getFitnessScore();
    std::cout << "[GlobalLocalizer] Prior ICP succeeded. fitness=" << m_last_fitness << std::endl;
    return icp.getFinalTransformation();
}

// ─── Full global localization ─────────────────────────────────────────────────
std::optional<Eigen::Matrix4f> GlobalLocalizer::localize(CloudType::Ptr scan,
                                                         const Eigen::Matrix3f &body_rot)
{
    if (!m_map_loaded || m_map_cloud->empty())
    {
        std::cerr << "[GlobalLocalizer] Map not loaded!" << std::endl;
        return std::nullopt;
    }

    // Downsample scan for coarse NDT scoring
    CloudType::Ptr scan_coarse(new CloudType);
    if (m_cfg.scan_voxel_coarse > 0)
    {
        m_voxel.setLeafSize(m_cfg.scan_voxel_coarse, m_cfg.scan_voxel_coarse, m_cfg.scan_voxel_coarse);
        m_voxel.setInputCloud(scan);
        m_voxel.filter(*scan_coarse);
    }
    else
    {
        scan_coarse = scan;
    }

    // Compute map centroid for Z anchoring of candidates
    Eigen::Vector4f map_centroid_h;
    pcl::compute3DCentroid(*m_map_coarse, map_centroid_h);
    Eigen::Vector3f map_centroid = map_centroid_h.head<3>();

    auto candidates = generateCandidates(map_centroid, body_rot);
    if (candidates.empty())
    {
        std::cerr << "[GlobalLocalizer] No search candidates generated "
                     "(degenerate map bounds?). Aborting search." << std::endl;
        return std::nullopt;
    }
    std::cout << "[GlobalLocalizer] Scoring " << candidates.size() << " candidates..." << std::endl;

    // Score each candidate: pre-transform scan, run NDT, get fitness + refined pose
    for (auto &c : candidates)
    {
        auto [score, refined] = scoreNDT(scan_coarse, c.transform);
        c.ndt_score = score;
        c.transform = refined;  // Use NDT-refined position as ICP starting point
    }

    // Keep top-K by NDT score
    std::sort(candidates.begin(), candidates.end(),
              [](const PoseCandidate &a, const PoseCandidate &b) {
                  return a.ndt_score < b.ndt_score;
              });
    int k = std::min(m_cfg.top_k, static_cast<int>(candidates.size()));
    candidates.resize(k);

    // Pre-filter: discard candidates whose NDT score exceeds the threshold.
    // Always keep at least top-1 as a fallback so we never send zero candidates to ICP.
    if (m_cfg.ndt_score_thresh > 0 && candidates.size() > 1)
    {
        auto cut = std::remove_if(candidates.begin() + 1, candidates.end(),
            [this](const PoseCandidate &c) { return c.ndt_score > m_cfg.ndt_score_thresh; });
        int removed = static_cast<int>(std::distance(cut, candidates.end()));
        if (removed > 0)
        {
            candidates.erase(cut, candidates.end());
            std::cout << "[GlobalLocalizer] NDT pre-filter: dropped " << removed
                      << " weak candidates (score > " << m_cfg.ndt_score_thresh << ")" << std::endl;
        }
    }

    std::cout << "[GlobalLocalizer] Best NDT score: " << candidates[0].ndt_score
              << " (" << candidates.size() << " candidates to ICP)" << std::endl;

    // Downsample for ICP refinement
    CloudType::Ptr scan_refine(new CloudType);
    if (m_cfg.scan_voxel_refine > 0)
    {
        m_voxel.setLeafSize(m_cfg.scan_voxel_refine, m_cfg.scan_voxel_refine, m_cfg.scan_voxel_refine);
        m_voxel.setInputCloud(scan);
        m_voxel.filter(*scan_refine);
    }
    else
    {
        scan_refine = scan;
    }

    // Refine top-K with full ICP, keep the best
    PoseCandidate best;
    best.icp_fitness = 1e9;
    for (int i = 0; i < k; i++)
    {
        PoseCandidate refined = refineICP(scan_refine, candidates[i].transform);
        std::cout << "[GlobalLocalizer] Candidate " << i
                  << " ICP fitness: " << refined.icp_fitness << std::endl;
        if (refined.icp_fitness < best.icp_fitness)
            best = refined;
    }

    if (best.icp_fitness > m_cfg.icp_fitness_thresh)
    {
        std::cout << "[GlobalLocalizer] Global localization FAILED. Best fitness: "
                  << best.icp_fitness << " > " << m_cfg.icp_fitness_thresh << std::endl;
        return std::nullopt;
    }

    m_last_fitness = best.icp_fitness;
    std::cout << "[GlobalLocalizer] SUCCESS. ICP fitness: " << m_last_fitness << std::endl;
    return best.transform;
}
