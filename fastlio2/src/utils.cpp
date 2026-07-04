#include "utils.h"
#include <cstring>
pcl::PointCloud<pcl::PointXYZINormal>::Ptr Utils::livox2PCL(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg, int filter_num, double min_range, double max_range)
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);
    int point_num = msg->point_num;
    cloud->reserve(point_num / filter_num + 1);
    for (int i = 0; i < point_num; i += filter_num)
    {
        if ((msg->points[i].line < 4) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
        {

            float x = msg->points[i].x;
            float y = msg->points[i].y;
            float z = msg->points[i].z;
            if (x * x + y * y + z * z < min_range * min_range || x * x + y * y + z * z > max_range * max_range)
                continue;
            pcl::PointXYZINormal p;
            p.x = x;
            p.y = y;
            p.z = z;
            p.intensity = msg->points[i].reflectivity;
            p.curvature = msg->points[i].offset_time / 1000000.0f;
            cloud->push_back(p);
        }
    }
    return cloud;
}

// Read one numeric field as float regardless of its wire datatype.
// memcpy avoids unaligned-access UB on the packed point buffer.
static float readFieldAsFloat(const uint8_t *ptr, uint8_t datatype)
{
    using sensor_msgs::msg::PointField;
    switch (datatype)
    {
    case PointField::FLOAT32: { float v;    std::memcpy(&v, ptr, 4); return v; }
    case PointField::FLOAT64: { double v;   std::memcpy(&v, ptr, 8); return static_cast<float>(v); }
    case PointField::UINT8:   { return static_cast<float>(*ptr); }
    case PointField::INT8:    { int8_t v;   std::memcpy(&v, ptr, 1); return v; }
    case PointField::UINT16:  { uint16_t v; std::memcpy(&v, ptr, 2); return v; }
    case PointField::INT16:   { int16_t v;  std::memcpy(&v, ptr, 2); return v; }
    case PointField::UINT32:  { uint32_t v; std::memcpy(&v, ptr, 4); return static_cast<float>(v); }
    case PointField::INT32:   { int32_t v;  std::memcpy(&v, ptr, 4); return static_cast<float>(v); }
    default: return 0.0f;
    }
}

pcl::PointCloud<pcl::PointXYZINormal>::Ptr Utils::pc2ToPCL(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg,
    int filter_num, double min_range, double max_range)
{
    using sensor_msgs::msg::PointField;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>);

    // Resolve field offsets + datatypes from the message descriptor.
    // Per-point time conventions differ by vendor:
    //   Velodyne: "time"      FLOAT32, seconds relative to header stamp
    //   Ouster:   "t"         UINT32,  nanoseconds relative to frame start
    //   Hesai:    "timestamp" FLOAT64, absolute UNIX seconds
    enum class TimeConv { NONE, REL_SEC, REL_NSEC, ABS_SEC };
    int x_off = -1, y_off = -1, z_off = -1, intensity_off = -1, time_off = -1;
    uint8_t intensity_type = 0, time_type = 0;
    TimeConv time_conv = TimeConv::NONE;

    for (const auto & f : msg->fields) {
        if      (f.name == "x" && f.datatype == PointField::FLOAT32) x_off = static_cast<int>(f.offset);
        else if (f.name == "y" && f.datatype == PointField::FLOAT32) y_off = static_cast<int>(f.offset);
        else if (f.name == "z" && f.datatype == PointField::FLOAT32) z_off = static_cast<int>(f.offset);
        else if (f.name == "intensity")
        {
            intensity_off  = static_cast<int>(f.offset);
            intensity_type = f.datatype;
        }
        else if (f.name == "time" || f.name == "time_stamp")
        {
            time_off  = static_cast<int>(f.offset);
            time_type = f.datatype;
            time_conv = TimeConv::REL_SEC;
        }
        else if (f.name == "t" && f.datatype == PointField::UINT32)
        {
            time_off  = static_cast<int>(f.offset);
            time_type = f.datatype;
            time_conv = TimeConv::REL_NSEC;
        }
        else if (f.name == "timestamp")
        {
            time_off  = static_cast<int>(f.offset);
            time_type = f.datatype;
            time_conv = (f.datatype == PointField::FLOAT64) ? TimeConv::ABS_SEC
                                                            : TimeConv::REL_SEC;
        }
    }
    if (x_off < 0 || y_off < 0 || z_off < 0) return cloud;

    const double header_sec = Utils::getSec(msg->header);
    const uint32_t point_step = msg->point_step;
    const uint32_t num_points = msg->width * msg->height;
    cloud->reserve(num_points / filter_num + 1);

    const double min_r2 = min_range * min_range;
    const double max_r2 = max_range * max_range;

    for (uint32_t i = 0; i < num_points; i += static_cast<uint32_t>(filter_num)) {
        const uint8_t * ptr = msg->data.data() + i * point_step;
        float x, y, z;
        std::memcpy(&x, ptr + x_off, 4);
        std::memcpy(&y, ptr + y_off, 4);
        std::memcpy(&z, ptr + z_off, 4);

        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
        double r2 = static_cast<double>(x)*x + static_cast<double>(y)*y + static_cast<double>(z)*z;
        if (r2 < min_r2 || r2 > max_r2) continue;

        pcl::PointXYZINormal p;
        p.x = x; p.y = y; p.z = z;
        p.intensity = (intensity_off >= 0)
            ? readFieldAsFloat(ptr + intensity_off, intensity_type) : 0.0f;

        // curvature stores the per-point time offset in milliseconds, consistent
        // with the livox2PCL convention used in syncPackage().
        float t_ms = 0.0f;
        if (time_off >= 0)
        {
            switch (time_conv)
            {
            case TimeConv::REL_SEC:
                t_ms = readFieldAsFloat(ptr + time_off, time_type) * 1000.0f;
                break;
            case TimeConv::REL_NSEC:
                t_ms = readFieldAsFloat(ptr + time_off, time_type) * 1e-6f;
                break;
            case TimeConv::ABS_SEC:
            {
                double abs_sec;
                std::memcpy(&abs_sec, ptr + time_off, 8);
                t_ms = static_cast<float>((abs_sec - header_sec) * 1000.0);
                break;
            }
            case TimeConv::NONE:
                break;
            }
        }
        p.curvature = t_ms;
        cloud->push_back(p);
    }
    return cloud;
}

double Utils::getSec(std_msgs::msg::Header &header)
{
    return static_cast<double>(header.stamp.sec) + static_cast<double>(header.stamp.nanosec) * 1e-9;
}
builtin_interfaces::msg::Time Utils::getTime(const double &sec)
{
    builtin_interfaces::msg::Time time_msg;
    time_msg.sec = static_cast<int32_t>(sec);
    time_msg.nanosec = static_cast<uint32_t>((sec - time_msg.sec) * 1e9);
    return time_msg;
}
