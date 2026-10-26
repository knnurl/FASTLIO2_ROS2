#include "commons.h"
bool esti_plane(PointVec &points, const double &thresh, V4D &out)
{
    // Stack storage (at most kMaxPlanePoints rows): this runs for every point in every
    // iteration, and heap-allocated MatrixXd made malloc/free a visible share of the CPU.
    if (points.size() > static_cast<size_t>(kMaxPlanePoints))
        return false;
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::ColMajor, kMaxPlanePoints, 3> A(points.size(), 3);
    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, kMaxPlanePoints, 1> b(points.size(), 1);
    A.setZero();
    b.setOnes();
    b *= -1.0;
    for (size_t i = 0; i < points.size(); i++)
    {
        A(i, 0) = points[i].x;
        A(i, 1) = points[i].y;
        A(i, 2) = points[i].z;
    }
    V3D normvec = A.colPivHouseholderQr().solve(b);
    double norm = normvec.norm();
    out[0] = normvec(0) / norm;
    out[1] = normvec(1) / norm;
    out[2] = normvec(2) / norm;
    out[3] = 1.0 / norm;
    for (size_t j = 0; j < points.size(); j++)
    {
        if (std::fabs(out(0) * points[j].x + out(1) * points[j].y + out(2) * points[j].z + out(3)) > thresh)
        {
            return false;
        }
    }
    return true;
}

float sq_dist(const PointType &p1, const PointType &p2)
{
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
}