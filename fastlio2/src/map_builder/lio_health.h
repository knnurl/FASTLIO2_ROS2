#pragma once
// Health of one LiDAR update (roadmap B1). Eigen only, no ROS, no PCL: unit tested in
// test/test_lio_health.cpp.
#include <Eigen/Eigen>
#include <cmath>

struct LioHealth
{
    bool valid = false;
    int points = 0;
    int matched = 0;
    double match_ratio = 0.0;
    double residual_rms = 0.0;
    int iterations = 0;
    // 3 * smallest eigenvalue / trace of the 3x3 information block: 1 = evenly constrained,
    // 0 = one direction unconstrained.
    double translation_degeneracy = 0.0;
    double rotation_degeneracy = 0.0;
    Eigen::Vector3d weakest_translation_dir = Eigen::Vector3d::Zero();
};

// 3 * lambda_min / trace of a symmetric positive semi-definite 3x3 block, and the eigenvector
// of lambda_min. 0 (and a zero vector) for an empty or non-finite block.
inline double conditioning(const Eigen::Matrix3d &block, Eigen::Vector3d *weakest = nullptr)
{
    if (weakest)
        weakest->setZero();
    const double trace = block.trace();
    if (!(trace > 0.0) || !std::isfinite(trace))
        return 0.0;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(block);
    if (solver.info() != Eigen::Success)
        return 0.0;
    if (weakest)
        *weakest = solver.eigenvectors().col(0);          // eigenvalues ascend
    const double value = 3.0 * std::max(0.0, solver.eigenvalues()(0)) / trace;
    return std::min(1.0, value);
}

// H: the measurement information matrix of the last iteration, state order
// [rotation(3), translation(3), ...] as in SharedState::H. sum_sq_residual: sum of squared
// point-to-plane distances of the matched points.
template <typename Derived>
LioHealth computeLioHealth(const Eigen::MatrixBase<Derived> &H, int points, int matched,
                           double sum_sq_residual, int iterations)
{
    LioHealth h;
    h.points = points;
    h.matched = matched;
    h.iterations = iterations;
    h.valid = matched > 0;
    if (points > 0)
        h.match_ratio = static_cast<double>(matched) / points;
    if (!h.valid)
        return h;
    h.residual_rms = std::sqrt(std::max(0.0, sum_sq_residual) / matched);
    const Eigen::Matrix3d rot = H.template block<3, 3>(0, 0);
    const Eigen::Matrix3d trans = H.template block<3, 3>(3, 3);
    h.rotation_degeneracy = conditioning(rot);
    h.translation_degeneracy = conditioning(trans, &h.weakest_translation_dir);
    return h;
}
