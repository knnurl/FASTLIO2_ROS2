#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "utils.h"

using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

namespace
{

struct Field
{
    std::string name;
    uint8_t datatype;
};

size_t sizeOf(uint8_t datatype)
{
    switch (datatype)
    {
    case PointField::INT8:
    case PointField::UINT8: return 1;
    case PointField::INT16:
    case PointField::UINT16: return 2;
    case PointField::FLOAT64: return 8;
    default: return 4;
    }
}

// Builds a one-row cloud with the given fields packed in order; each point is
// a list of values, one per field, written in that field's datatype.
PointCloud2::SharedPtr makeCloud(const std::vector<Field> &fields,
                                 const std::vector<std::vector<double>> &points,
                                 int32_t stamp_sec = 0, uint32_t stamp_nanosec = 0)
{
    auto msg = std::make_shared<PointCloud2>();
    msg->header.stamp.sec = stamp_sec;
    msg->header.stamp.nanosec = stamp_nanosec;
    uint32_t offset = 0;
    for (const auto &f : fields)
    {
        PointField pf;
        pf.name = f.name;
        pf.offset = offset;
        pf.datatype = f.datatype;
        pf.count = 1;
        msg->fields.push_back(pf);
        offset += static_cast<uint32_t>(sizeOf(f.datatype));
    }
    msg->point_step = offset;
    msg->height = 1;
    msg->width = static_cast<uint32_t>(points.size());
    msg->row_step = msg->point_step * msg->width;
    msg->data.resize(msg->row_step);
    for (size_t i = 0; i < points.size(); ++i)
    {
        uint8_t *ptr = msg->data.data() + i * msg->point_step;
        for (size_t j = 0; j < fields.size(); ++j)
        {
            uint8_t *dst = ptr + msg->fields[j].offset;
            const double v = points[i][j];
            switch (fields[j].datatype)
            {
            case PointField::FLOAT32: { float x = static_cast<float>(v); std::memcpy(dst, &x, 4); break; }
            case PointField::FLOAT64: { std::memcpy(dst, &v, 8); break; }
            case PointField::UINT8:   { uint8_t x = static_cast<uint8_t>(v); std::memcpy(dst, &x, 1); break; }
            case PointField::UINT16:  { uint16_t x = static_cast<uint16_t>(v); std::memcpy(dst, &x, 2); break; }
            case PointField::UINT32:  { uint32_t x = static_cast<uint32_t>(v); std::memcpy(dst, &x, 4); break; }
            default: ADD_FAILURE() << "unsupported test datatype";
            }
        }
    }
    return msg;
}

const Field X{"x", PointField::FLOAT32};
const Field Y{"y", PointField::FLOAT32};
const Field Z{"z", PointField::FLOAT32};

}  // namespace

TEST(Pc2ToPcl, VelodyneRelativeSecondsBecomeMilliseconds)
{
    auto msg = makeCloud({X, Y, Z, {"intensity", PointField::FLOAT32}, {"ring", PointField::UINT16},
                          {"time", PointField::FLOAT32}},
                         {{1.0, 0.0, 0.0, 5.0, 3.0, 0.05}, {0.0, 2.0, 0.0, 7.0, 4.0, 0.075}});
    auto cloud = Utils::pc2ToPCL(msg, 1);
    ASSERT_EQ(cloud->size(), 2u);
    EXPECT_FLOAT_EQ(cloud->points[0].x, 1.0f);
    EXPECT_FLOAT_EQ(cloud->points[0].intensity, 5.0f);
    EXPECT_NEAR(cloud->points[0].curvature, 50.0f, 1e-3);
    EXPECT_FLOAT_EQ(cloud->points[1].y, 2.0f);
    EXPECT_NEAR(cloud->points[1].curvature, 75.0f, 1e-3);
}

TEST(Pc2ToPcl, OusterNanosecondsBecomeMilliseconds)
{
    auto msg = makeCloud({X, Y, Z, {"t", PointField::UINT32}, {"intensity", PointField::UINT16}},
                         {{3.0, 0.0, 0.0, 25000000.0, 300.0}});
    auto cloud = Utils::pc2ToPCL(msg, 1);
    ASSERT_EQ(cloud->size(), 1u);
    EXPECT_NEAR(cloud->points[0].curvature, 25.0f, 1e-3);
    EXPECT_FLOAT_EQ(cloud->points[0].intensity, 300.0f);
}

TEST(Pc2ToPcl, HesaiAbsoluteTimestampIsRelativeToTheHeader)
{
    auto msg = makeCloud({X, Y, Z, {"intensity", PointField::UINT8}, {"timestamp", PointField::FLOAT64}},
                         {{0.0, 0.0, 4.0, 200.0, 1700000000.010}},
                         1700000000, 0);
    auto cloud = Utils::pc2ToPCL(msg, 1);
    ASSERT_EQ(cloud->size(), 1u);
    EXPECT_NEAR(cloud->points[0].curvature, 10.0f, 1e-2);
    EXPECT_FLOAT_EQ(cloud->points[0].intensity, 200.0f);
}

TEST(Pc2ToPcl, NoTimeOrIntensityFieldsGiveZero)
{
    auto msg = makeCloud({X, Y, Z}, {{1.0, 1.0, 1.0}});
    auto cloud = Utils::pc2ToPCL(msg, 1);
    ASSERT_EQ(cloud->size(), 1u);
    EXPECT_FLOAT_EQ(cloud->points[0].curvature, 0.0f);
    EXPECT_FLOAT_EQ(cloud->points[0].intensity, 0.0f);
}

TEST(Pc2ToPcl, DropsPointsOutsideRangeAndNonFinite)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    auto msg = makeCloud({X, Y, Z},
                         {{0.2, 0.0, 0.0}, {5.0, 0.0, 0.0}, {20.0, 0.0, 0.0}, {nan, 0.0, 0.0}});
    auto cloud = Utils::pc2ToPCL(msg, 1, 0.5, 10.0);
    ASSERT_EQ(cloud->size(), 1u);
    EXPECT_FLOAT_EQ(cloud->points[0].x, 5.0f);
}

TEST(Pc2ToPcl, FilterNumKeepsEveryNthPoint)
{
    auto msg = makeCloud({X, Y, Z},
                         {{1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}, {5, 0, 0}});
    auto cloud = Utils::pc2ToPCL(msg, 2);
    ASSERT_EQ(cloud->size(), 3u);
    EXPECT_FLOAT_EQ(cloud->points[0].x, 1.0f);
    EXPECT_FLOAT_EQ(cloud->points[1].x, 3.0f);
    EXPECT_FLOAT_EQ(cloud->points[2].x, 5.0f);
}

TEST(Pc2ToPcl, MissingCoordinateFieldGivesAnEmptyCloud)
{
    auto msg = makeCloud({X, Y, {"intensity", PointField::FLOAT32}}, {{1.0, 1.0, 1.0}});
    EXPECT_TRUE(Utils::pc2ToPCL(msg, 1)->empty());
}
