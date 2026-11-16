// Roadmap B1: the degeneracy measure on scenes whose answer is known.
#include <gtest/gtest.h>
#include <vector>
#include "map_builder/lio_health.h"

namespace
{
using V3 = Eigen::Vector3d;
using M12 = Eigen::Matrix<double, 12, 12>;

// The filter's measurement information for points p (body = world, identity pose) lying on
// planes with unit normals n: J = [-(n^T) [p]x , n^T], H = sum J^T J * w.
M12 information(const std::vector<V3> &normals, const std::vector<V3> &points, double w = 1000.0)
{
    M12 H = M12::Zero();
    for (size_t i = 0; i < normals.size(); i++)
    {
        Eigen::Matrix<double, 1, 12> J = Eigen::Matrix<double, 1, 12>::Zero();
        const V3 &n = normals[i], &p = points[i];
        Eigen::Matrix3d hat;
        hat << 0, -p.z(), p.y(), p.z(), 0, -p.x(), -p.y(), p.x(), 0;
        J.block<1, 3>(0, 0) = -n.transpose() * hat;
        J.block<1, 3>(0, 3) = n.transpose();
        H += J.transpose() * w * J;
    }
    return H;
}

void wall(std::vector<V3> &n, std::vector<V3> &p, const V3 &normal, const V3 &origin, const V3 &u, const V3 &v)
{
    for (double a = -2.0; a <= 2.0; a += 0.5)
        for (double b = -1.0; b <= 1.0; b += 0.5)
        {
            n.push_back(normal);
            p.push_back(origin + a * u + b * v);
        }
}
} // namespace

TEST(LioHealth, RoomConstrainsEveryDirection)
{
    std::vector<V3> n, p;
    wall(n, p, V3::UnitX(), V3(3, 0, 1), V3::UnitY(), V3::UnitZ());
    wall(n, p, -V3::UnitX(), V3(-3, 0, 1), V3::UnitY(), V3::UnitZ());
    wall(n, p, V3::UnitY(), V3(0, 3, 1), V3::UnitX(), V3::UnitZ());
    wall(n, p, V3::UnitZ(), V3(0, 0, -1), V3::UnitX(), V3::UnitY());
    LioHealth h = computeLioHealth(information(n, p), 400, static_cast<int>(n.size()), 0.0, 3);
    EXPECT_TRUE(h.valid);
    EXPECT_GT(h.translation_degeneracy, 0.6);
    EXPECT_GT(h.rotation_degeneracy, 0.1);
}

TEST(LioHealth, CorridorIsWeakAlongItsAxis)
{
    // Two side walls, floor and ceiling; nothing with a normal along x.
    std::vector<V3> n, p;
    wall(n, p, V3::UnitY(), V3(0, 1, 1), V3::UnitX(), V3::UnitZ());
    wall(n, p, -V3::UnitY(), V3(0, -1, 1), V3::UnitX(), V3::UnitZ());
    wall(n, p, V3::UnitZ(), V3(0, 0, 0), V3::UnitX(), V3::UnitY());
    wall(n, p, -V3::UnitZ(), V3(0, 0, 2.5), V3::UnitX(), V3::UnitY());
    LioHealth h = computeLioHealth(information(n, p), 400, static_cast<int>(n.size()), 0.0, 3);
    EXPECT_LT(h.translation_degeneracy, 1e-6);
    EXPECT_NEAR(std::fabs(h.weakest_translation_dir.x()), 1.0, 1e-6);
}

TEST(LioHealth, FarEndWallRestoresTheCorridor)
{
    std::vector<V3> n, p;
    wall(n, p, V3::UnitY(), V3(0, 1, 1), V3::UnitX(), V3::UnitZ());
    wall(n, p, -V3::UnitY(), V3(0, -1, 1), V3::UnitX(), V3::UnitZ());
    wall(n, p, V3::UnitZ(), V3(0, 0, 0), V3::UnitX(), V3::UnitY());
    LioHealth weak = computeLioHealth(information(n, p), 300, static_cast<int>(n.size()), 0.0, 3);
    for (int k = 0; k < 6; k++)                      // a few returns from an end wall
    {
        n.push_back(-V3::UnitX());
        p.push_back(V3(8, -0.5 + 0.2 * k, 1));
    }
    LioHealth better = computeLioHealth(information(n, p), 300, static_cast<int>(n.size()), 0.0, 3);
    EXPECT_GT(better.translation_degeneracy, weak.translation_degeneracy + 0.05);
}

TEST(LioHealth, SingleWallIsWeakInTwoDirections)
{
    std::vector<V3> n, p;
    wall(n, p, V3::UnitX(), V3(3, 0, 1), V3::UnitY(), V3::UnitZ());
    LioHealth h = computeLioHealth(information(n, p), 100, static_cast<int>(n.size()), 0.0, 3);
    EXPECT_LT(h.translation_degeneracy, 1e-6);
    EXPECT_NEAR(h.weakest_translation_dir.x(), 0.0, 1e-6);   // weak within the wall plane
}

TEST(LioHealth, MatchRatioResidualAndInvalid)
{
    std::vector<V3> n, p;
    wall(n, p, V3::UnitX(), V3(3, 0, 1), V3::UnitY(), V3::UnitZ());
    const int m = static_cast<int>(n.size());
    LioHealth h = computeLioHealth(information(n, p), 2 * m, m, m * 0.02 * 0.02, 2);
    EXPECT_NEAR(h.match_ratio, 0.5, 1e-12);
    EXPECT_NEAR(h.residual_rms, 0.02, 1e-12);
    EXPECT_EQ(h.iterations, 2);

    LioHealth none = computeLioHealth(M12::Zero(), 500, 0, 0.0, 0);
    EXPECT_FALSE(none.valid);
    EXPECT_EQ(none.translation_degeneracy, 0.0);
    EXPECT_EQ(none.match_ratio, 0.0);
    LioHealth empty = computeLioHealth(M12::Zero(), 0, 0, 0.0, 0);
    EXPECT_FALSE(empty.valid);
}

TEST(LioHealth, ConditioningIsScaleFreeAndBounded)
{
    Eigen::Matrix3d a = Eigen::Vector3d(4, 4, 4).asDiagonal();
    EXPECT_NEAR(conditioning(a), 1.0, 1e-12);
    EXPECT_NEAR(conditioning(1e6 * a), 1.0, 1e-12);
    Eigen::Matrix3d nan = Eigen::Matrix3d::Constant(std::nan(""));
    EXPECT_EQ(conditioning(nan), 0.0);
    EXPECT_EQ(conditioning(Eigen::Matrix3d::Zero()), 0.0);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
