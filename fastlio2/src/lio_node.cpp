#include <mutex>
#include <vector>
#include <queue>
#include <memory>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <pcl/io/pcd_io.h>
#include "interface/srv/save_maps.hpp"

#include "utils.h"
#include "map_builder/commons.h"
#include "map_builder/map_builder.h"

#include <pcl_conversions/pcl_conversions.h>
#include "tf2_ros/transform_broadcaster.h"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <yaml-cpp/yaml.h>

using namespace std::chrono_literals;

// Read a required YAML key, failing with the key name instead of a bare
// yaml-cpp exception that doesn't say what is missing.
template <typename T>
static T yamlRequired(const YAML::Node &config, const std::string &key)
{
    if (!config[key])
        throw std::runtime_error("missing required config key '" + key + "'");
    try
    {
        return config[key].as<T>();
    }
    catch (const YAML::Exception &)
    {
        throw std::runtime_error("config key '" + key + "' has the wrong type");
    }
}

// Read an optional YAML key; absent means the default. A present key of the wrong type still fails.
template <typename T>
static T yamlOptional(const YAML::Node &config, const std::string &key, const T &fallback)
{
    return config[key] ? yamlRequired<T>(config, key) : fallback;
}

struct NodeConfig
{
    std::string imu_topic = "/livox/imu";
    std::string lidar_topic = "/livox/lidar";
    std::string body_frame = "body";
    std::string world_frame = "lidar";
    bool print_time_cost = false;
    // Optional outputs. Off means the work is not done at all (no transform, no accumulation), which
    // matters on small computers: the accumulated map alone costs ~250 MB. Defaults keep the
    // previous behaviour. /lio_odom and the TF are always published.
    bool publish_body_cloud = true;
    bool publish_world_cloud = true;
    bool publish_accumulated_map = true;
    bool publish_path = true;
    // 0 = Livox CustomMsg (default), 1 = standard PointCloud2 (Velodyne, Ouster, etc.)
    int lidar_type = 0;
    // Multiplier applied to IMU linear acceleration to get m/s².
    // Livox IMUs report acceleration in g → 10.0 (upstream convention).
    // Standard IMUs (e.g. Spot, most PointCloud2 setups) already report
    // m/s² → set acc_scale: 1.0 in the config.
    double acc_scale = 10.0;
};
struct StateData
{
    bool lidar_pushed = false;
    std::mutex imu_mutex;
    std::mutex lidar_mutex;
    double last_lidar_time = -1.0;
    double last_imu_time = -1.0;
    std::deque<IMUData> imu_buffer;
    std::deque<std::pair<double, pcl::PointCloud<pcl::PointXYZINormal>::Ptr>> lidar_buffer;
    nav_msgs::msg::Path path;
};

class LIONode : public rclcpp::Node
{
public:
    LIONode() : Node("lio_node")
    {
        RCLCPP_INFO(this->get_logger(), "LIO Node Started");
        loadParameters();

        // Sensor-data QoS (best-effort, volatile): compatible with both
        // reliable and best-effort driver publishers, and never back-pressures
        // the driver. IMU depth covers ~0.5 s at 200 Hz.
        rclcpp::QoS imu_qos   = rclcpp::SensorDataQoS().keep_last(100);
        rclcpp::QoS lidar_qos = rclcpp::SensorDataQoS().keep_last(10);
        m_imu_sub = this->create_subscription<sensor_msgs::msg::Imu>(m_node_config.imu_topic, imu_qos, std::bind(&LIONode::imuCB, this, std::placeholders::_1));
        if (m_node_config.lidar_type == 0) {
            m_lidar_sub = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                m_node_config.lidar_topic, lidar_qos, std::bind(&LIONode::lidarCB, this, std::placeholders::_1));
            RCLCPP_INFO(this->get_logger(), "Lidar type: Livox CustomMsg on %s", m_node_config.lidar_topic.c_str());
        } else {
            m_lidar_pc2_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                m_node_config.lidar_topic, lidar_qos, std::bind(&LIONode::lidarPC2CB, this, std::placeholders::_1));
            RCLCPP_INFO(this->get_logger(), "Lidar type: PointCloud2 on %s", m_node_config.lidar_topic.c_str());
        }

        // Reliable, shallow queues: consumers (localizer, RViz) read these live;
        // a deep history on multi-MB clouds is just a memory bomb if one stalls.
        m_body_cloud_pub  = this->create_publisher<sensor_msgs::msg::PointCloud2>("body_cloud", 10);
        m_world_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("world_cloud", 10);
        m_accum_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("accumulated_map", 10);
        m_path_pub = this->create_publisher<nav_msgs::msg::Path>("lio_path", 10);
        m_odom_pub = this->create_publisher<nav_msgs::msg::Odometry>("lio_odom", 10);
        m_tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

        m_state_data.path.poses.clear();
        m_state_data.path.header.frame_id = m_node_config.world_frame;

        m_accum_cloud.reset(new CloudType);
        float vox = static_cast<float>(m_builder_config.map_resolution);
        m_accum_voxel.setLeafSize(vox, vox, vox);

        m_kf = std::make_shared<IESKF>();
        m_builder = std::make_shared<MapBuilder>(m_builder_config, m_kf);
        m_timer      = this->create_wall_timer(20ms, std::bind(&LIONode::timerCB, this));
        if (m_node_config.publish_accumulated_map)
            m_accum_timer = this->create_wall_timer(1000ms, std::bind(&LIONode::publishAccumCloud, this));

        // The PCD save is slow (compressed write of the full map); it runs in
        // its own callback group so the MultiThreadedExecutor can keep the
        // odometry timer and sensor callbacks running while it works.
        m_service_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        m_save_map_srv = this->create_service<interface::srv::SaveMaps>(
            "save_map", std::bind(&LIONode::saveMapCB, this, std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, m_service_cb_group);
        RCLCPP_INFO(this->get_logger(), "SaveMap service ready at ~/save_map");
    }

    void saveMapCB(const std::shared_ptr<interface::srv::SaveMaps::Request> request,
                   std::shared_ptr<interface::srv::SaveMaps::Response> response)
    {
        // Copy the map points under the builder mutex (the timer thread mutates
        // the ikd-tree in process()), then do all file IO without any lock held.
        PointVec map_points;
        {
            std::lock_guard<std::mutex> lock(m_builder_mutex);
            if (m_builder->status() != BuilderStatus::MAPPING)
            {
                response->success = false;
                response->message = "LIO not in MAPPING state yet";
                return;
            }
            m_builder->lidar_processor()->getMapPoints(map_points);
        }
        if (map_points.empty())
        {
            response->success = false;
            response->message = "Map is empty";
            return;
        }
        std::string path = request->file_path;
        if (path.empty())
        {
            const char *home = std::getenv("HOME");
            path = std::string(home ? home : ".") + "/fastlio2_ws/maps/map.pcd";
        }

        // Ensure parent directory exists
        std::filesystem::path p(path);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());

        // If file already exists, rename to <stem>_YYYYMMDD_HHMMSS.pcd
        if (std::filesystem::exists(p))
        {
            auto now = std::chrono::system_clock::now();
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            std::tm tm_buf;
            localtime_r(&t, &tm_buf); // thread-safe; this runs on the service thread
            char buf[32];
            std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_buf);
            std::filesystem::path stamped =
                p.parent_path() / (p.stem().string() + "_" + buf + ".pcd");
            path = stamped.string();
            RCLCPP_INFO(this->get_logger(),
                "Map file already exists — saving as %s", path.c_str());
        }

        CloudType::Ptr cloud(new CloudType);
        cloud->points = map_points;
        cloud->width = static_cast<uint32_t>(map_points.size());
        cloud->height = 1;
        cloud->is_dense = false;

        int ret = pcl::io::savePCDFileBinaryCompressed(path, *cloud);
        if (ret < 0)
        {
            response->success = false;
            response->message = "Failed to write PCD to " + path;
        }
        else
        {
            response->success = true;
            response->message = "Saved " + std::to_string(map_points.size()) + " points to " + path;
            RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
        }
    }

    // Throws std::runtime_error with a message naming the file/key on any
    // config problem; main() turns that into a fatal log and clean exit.
    void loadParameters()
    {
        this->declare_parameter("config_path", "");
        std::string config_path;
        this->get_parameter<std::string>("config_path", config_path);

        if (config_path.empty())
            throw std::runtime_error(
                "config_path parameter is not set (pass -p config_path:=<lio.yaml>)");

        YAML::Node config;
        try
        {
            config = YAML::LoadFile(config_path);
        }
        catch (const YAML::Exception &e)
        {
            throw std::runtime_error(
                "failed to load YAML config '" + config_path + "': " + e.what());
        }

        RCLCPP_INFO(this->get_logger(), "LOAD FROM YAML CONFIG PATH: %s", config_path.c_str());

        m_node_config.imu_topic = yamlRequired<std::string>(config, "imu_topic");
        m_node_config.lidar_topic = yamlRequired<std::string>(config, "lidar_topic");
        m_node_config.body_frame = yamlRequired<std::string>(config, "body_frame");
        m_node_config.world_frame = yamlRequired<std::string>(config, "world_frame");
        m_node_config.print_time_cost = yamlRequired<bool>(config, "print_time_cost");
        m_node_config.publish_body_cloud = yamlOptional<bool>(config, "publish_body_cloud", true);
        m_node_config.publish_world_cloud = yamlOptional<bool>(config, "publish_world_cloud", true);
        m_node_config.publish_accumulated_map = yamlOptional<bool>(config, "publish_accumulated_map", true);
        m_node_config.publish_path = yamlOptional<bool>(config, "publish_path", true);
        if (config["lidar_type"]) {
            m_node_config.lidar_type = config["lidar_type"].as<int>();
        }
        if (config["acc_scale"]) {
            m_node_config.acc_scale = config["acc_scale"].as<double>();
        }
        else if (m_node_config.lidar_type != 0) {
            // Non-Livox setups usually pair a standard IMU reporting m/s²;
            // silently applying the Livox g→m/s² factor would inflate gravity
            // to ~100 m/s² and break initialization.
            RCLCPP_WARN(this->get_logger(),
                "lidar_type=%d but acc_scale is not set — defaulting to %.1f "
                "(Livox g→m/s²). If your IMU already reports m/s², set acc_scale: 1.0",
                m_node_config.lidar_type, m_node_config.acc_scale);
        }

        m_builder_config.lidar_filter_num = yamlRequired<int>(config, "lidar_filter_num");
        if (m_builder_config.lidar_filter_num < 1)
            throw std::runtime_error("lidar_filter_num must be >= 1");
        m_builder_config.lidar_min_range = yamlRequired<double>(config, "lidar_min_range");
        m_builder_config.lidar_max_range = yamlRequired<double>(config, "lidar_max_range");
        m_builder_config.scan_resolution = yamlRequired<double>(config, "scan_resolution");
        m_builder_config.map_resolution = yamlRequired<double>(config, "map_resolution");
        m_builder_config.cube_len = yamlRequired<double>(config, "cube_len");
        m_builder_config.det_range = yamlRequired<double>(config, "det_range");
        m_builder_config.move_thresh = yamlRequired<double>(config, "move_thresh");
        m_builder_config.na = yamlRequired<double>(config, "na");
        m_builder_config.ng = yamlRequired<double>(config, "ng");
        m_builder_config.nba = yamlRequired<double>(config, "nba");
        m_builder_config.nbg = yamlRequired<double>(config, "nbg");

        m_builder_config.imu_init_num = yamlRequired<int>(config, "imu_init_num");
        m_builder_config.near_search_num = yamlRequired<int>(config, "near_search_num");
        if (m_builder_config.near_search_num < 3 || m_builder_config.near_search_num > kMaxPlanePoints)
            throw std::runtime_error("near_search_num must be between 3 and " + std::to_string(kMaxPlanePoints));
        m_builder_config.ieskf_max_iter = yamlRequired<int>(config, "ieskf_max_iter");
        m_builder_config.gravity_align = yamlRequired<bool>(config, "gravity_align");
        m_builder_config.esti_il = yamlRequired<bool>(config, "esti_il");
        std::vector<double> t_il_vec = yamlRequired<std::vector<double>>(config, "t_il");
        std::vector<double> r_il_vec = yamlRequired<std::vector<double>>(config, "r_il");
        if (t_il_vec.size() != 3 || r_il_vec.size() != 9)
            throw std::runtime_error("t_il must have 3 elements and r_il 9 (row-major 3x3)");
        m_builder_config.t_il << t_il_vec[0], t_il_vec[1], t_il_vec[2];
        m_builder_config.r_il << r_il_vec[0], r_il_vec[1], r_il_vec[2], r_il_vec[3], r_il_vec[4], r_il_vec[5], r_il_vec[6], r_il_vec[7], r_il_vec[8];
        m_builder_config.lidar_cov_inv = yamlRequired<double>(config, "lidar_cov_inv");
    }

    void imuCB(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(m_state_data.imu_mutex);
        double timestamp = Utils::getSec(msg->header);
        if (timestamp < m_state_data.last_imu_time)
        {
            RCLCPP_WARN(this->get_logger(), "IMU Message is out of order");
            std::deque<IMUData>().swap(m_state_data.imu_buffer);
        }
        m_state_data.imu_buffer.emplace_back(V3D(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z) * m_node_config.acc_scale,
                                             V3D(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                                             timestamp);
        m_state_data.last_imu_time = timestamp;
    }
    void lidarCB(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg)
    {
        CloudType::Ptr cloud = Utils::livox2PCL(msg, m_builder_config.lidar_filter_num, m_builder_config.lidar_min_range, m_builder_config.lidar_max_range);
        // Range/tag filtering can leave nothing; syncPackage assumes non-empty
        // clouds (points.back()), so drop the scan here.
        if (cloud->points.empty())
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Lidar scan empty after filtering — dropping scan");
            return;
        }
        std::lock_guard<std::mutex> lock(m_state_data.lidar_mutex);
        double timestamp = Utils::getSec(msg->header);
        if (timestamp < m_state_data.last_lidar_time)
        {
            RCLCPP_WARN(this->get_logger(), "Lidar Message is out of order");
            std::deque<std::pair<double, pcl::PointCloud<pcl::PointXYZINormal>::Ptr>>().swap(m_state_data.lidar_buffer);
        }
        m_state_data.lidar_buffer.emplace_back(timestamp, cloud);
        m_state_data.last_lidar_time = timestamp;
    }

    void lidarPC2CB(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        CloudType::Ptr cloud = Utils::pc2ToPCL(msg, m_builder_config.lidar_filter_num, m_builder_config.lidar_min_range, m_builder_config.lidar_max_range);
        // Empty when x/y/z fields are missing or everything was range-filtered.
        if (cloud->points.empty())
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "PointCloud2 scan empty after conversion/filtering — dropping scan "
                "(check the topic's field layout if this persists)");
            return;
        }
        std::lock_guard<std::mutex> lock(m_state_data.lidar_mutex);
        double timestamp = Utils::getSec(msg->header);
        if (timestamp < m_state_data.last_lidar_time)
        {
            RCLCPP_WARN(this->get_logger(), "Lidar Message is out of order");
            std::deque<std::pair<double, pcl::PointCloud<pcl::PointXYZINormal>::Ptr>>().swap(m_state_data.lidar_buffer);
        }
        m_state_data.lidar_buffer.emplace_back(timestamp, cloud);
        m_state_data.last_lidar_time = timestamp;
    }

    bool syncPackage()
    {
        // Both buffers are filled by the sensor callbacks; hold both mutexes
        // for the whole sync so the package is assembled from a consistent view.
        std::lock_guard<std::mutex> lidar_lock(m_state_data.lidar_mutex);
        std::lock_guard<std::mutex> imu_lock(m_state_data.imu_mutex);
        if (m_state_data.imu_buffer.empty() || m_state_data.lidar_buffer.empty())
            return false;
        if (!m_state_data.lidar_pushed)
        {
            // Callbacks drop empty scans, but stay defensive: points.back()
            // below is undefined behaviour on an empty cloud.
            if (m_state_data.lidar_buffer.front().second->points.empty())
            {
                m_state_data.lidar_buffer.pop_front();
                return false;
            }
            m_package.cloud = m_state_data.lidar_buffer.front().second;
            std::sort(m_package.cloud->points.begin(), m_package.cloud->points.end(), [](PointType &p1, PointType &p2)
                      { return p1.curvature < p2.curvature; });
            m_package.cloud_start_time = m_state_data.lidar_buffer.front().first;
            m_package.cloud_end_time = m_package.cloud_start_time + m_package.cloud->points.back().curvature / 1000.0;
            m_state_data.lidar_pushed = true;
        }
        if (m_state_data.last_imu_time < m_package.cloud_end_time)
            return false;

        Vec<IMUData>().swap(m_package.imus);
        while (!m_state_data.imu_buffer.empty() && m_state_data.imu_buffer.front().time < m_package.cloud_end_time)
        {
            m_package.imus.emplace_back(m_state_data.imu_buffer.front());
            m_state_data.imu_buffer.pop_front();
        }
        m_state_data.lidar_buffer.pop_front();
        m_state_data.lidar_pushed = false;
        return true;
    }

    void publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub, CloudType::Ptr cloud, const std::string &frame_id, const double &time)
    {
        if (pub->get_subscription_count() <= 0)
            return;
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.frame_id = frame_id;
        cloud_msg.header.stamp = Utils::getTime(time);
        pub->publish(cloud_msg);
    }

    void publishOdometry(rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub, const std::string &frame_id, const std::string &child_frame, const double &time)
    {
        if (odom_pub->get_subscription_count() <= 0)
            return;
        nav_msgs::msg::Odometry odom;
        odom.header.frame_id = frame_id;
        odom.header.stamp = Utils::getTime(time);
        odom.child_frame_id = child_frame;
        odom.pose.pose.position.x = m_kf->x().t_wi.x();
        odom.pose.pose.position.y = m_kf->x().t_wi.y();
        odom.pose.pose.position.z = m_kf->x().t_wi.z();
        Eigen::Quaterniond q(m_kf->x().r_wi);
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();

        V3D vel = m_kf->x().r_wi.transpose() * m_kf->x().v;
        odom.twist.twist.linear.x = vel.x();
        odom.twist.twist.linear.y = vel.y();
        odom.twist.twist.linear.z = vel.z();
        odom_pub->publish(odom);
    }

    void publishPath(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub, const std::string &frame_id, const double &time)
    {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = frame_id;
        pose.header.stamp = Utils::getTime(time);
        pose.pose.position.x = m_kf->x().t_wi.x();
        pose.pose.position.y = m_kf->x().t_wi.y();
        pose.pose.position.z = m_kf->x().t_wi.z();
        Eigen::Quaterniond q(m_kf->x().r_wi);
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
        // Record the pose whether or not anyone is subscribed, so the path has
        // no holes when RViz attaches late.
        m_state_data.path.poses.push_back(pose);

        // The path grows for the whole session and is re-sent in full on every
        // scan; unbounded it is both a memory leak and O(n²) bandwidth. When it
        // hits the cap, drop every other pose — the trajectory keeps its shape,
        // spacing just gets coarser the longer the run.
        constexpr size_t MAX_PATH_POSES = 10000;
        if (m_state_data.path.poses.size() >= MAX_PATH_POSES)
        {
            auto &poses = m_state_data.path.poses;
            size_t keep = 0;
            for (size_t i = 0; i < poses.size(); i += 2)
                poses[keep++] = poses[i];
            poses.resize(keep);
        }

        if (path_pub->get_subscription_count() <= 0)
            return;
        path_pub->publish(m_state_data.path);
    }

    void broadCastTF(std::shared_ptr<tf2_ros::TransformBroadcaster> broad_caster, const std::string &frame_id, const std::string &child_frame, const double &time)
    {
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.frame_id = frame_id;
        transformStamped.child_frame_id = child_frame;
        transformStamped.header.stamp = Utils::getTime(time);
        Eigen::Quaterniond q(m_kf->x().r_wi);
        V3D t = m_kf->x().t_wi;
        transformStamped.transform.translation.x = t.x();
        transformStamped.transform.translation.y = t.y();
        transformStamped.transform.translation.z = t.z();
        transformStamped.transform.rotation.x = q.x();
        transformStamped.transform.rotation.y = q.y();
        transformStamped.transform.rotation.z = q.z();
        transformStamped.transform.rotation.w = q.w();
        broad_caster->sendTransform(transformStamped);
    }

    void timerCB()
    {
        if (!syncPackage())
            return;
        auto t1 = std::chrono::high_resolution_clock::now();
        BuilderStatus status;
        {
            std::lock_guard<std::mutex> lock(m_builder_mutex);
            m_builder->process(m_package);
            status = m_builder->status();
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        if (m_node_config.print_time_cost)
        {
            auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;
            RCLCPP_WARN(this->get_logger(), "Time cost: %.2f ms", time_used);
        }

        if (status != BuilderStatus::MAPPING)
            return;

        broadCastTF(m_tf_broadcaster, m_node_config.world_frame, m_node_config.body_frame, m_package.cloud_end_time);

        publishOdometry(m_odom_pub, m_node_config.world_frame, m_node_config.body_frame, m_package.cloud_end_time);

        // Clouds are transformed only when something will use them: a subscriber now, or the
        // accumulated map. The transforms copy the full-resolution scan.
        if (m_node_config.publish_body_cloud && m_body_cloud_pub->get_subscription_count() > 0)
        {
            CloudType::Ptr body_cloud = m_builder->lidar_processor()->transformCloud(m_package.cloud, m_kf->x().r_il, m_kf->x().t_il);
            publishCloud(m_body_cloud_pub, body_cloud, m_node_config.body_frame, m_package.cloud_end_time);
        }

        const bool world_cloud_wanted = m_node_config.publish_world_cloud && m_world_cloud_pub->get_subscription_count() > 0;
        if (world_cloud_wanted || m_node_config.publish_accumulated_map)
        {
            CloudType::Ptr world_cloud = m_builder->lidar_processor()->transformCloud(m_package.cloud, m_builder->lidar_processor()->r_wl(), m_builder->lidar_processor()->t_wl());
            if (world_cloud_wanted)
                publishCloud(m_world_cloud_pub, world_cloud, m_node_config.world_frame, m_package.cloud_end_time);

            if (m_node_config.publish_accumulated_map)
            {
                // Accumulate into the persistent map cloud (published at 1 Hz).
                // Voxel-filter when the cloud exceeds 2 M points to keep memory and
                // serialisation time bounded; map_resolution voxels match the ikd-tree.
                *m_accum_cloud += *world_cloud;
                m_accum_last_time = m_package.cloud_end_time;
                if (m_accum_cloud->size() > 2000000)
                {
                    CloudType::Ptr filtered(new CloudType);
                    m_accum_voxel.setInputCloud(m_accum_cloud);
                    m_accum_voxel.filter(*filtered);
                    m_accum_cloud = filtered;
                }
            }
        }

        if (m_node_config.publish_path)
            publishPath(m_path_pub, m_node_config.world_frame, m_package.cloud_end_time);
    }

    void publishAccumCloud()
    {
        if (!m_accum_cloud || m_accum_cloud->empty())
            return;
        if (m_accum_cloud_pub->get_subscription_count() < 1)
            return;
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*m_accum_cloud, msg);
        msg.header.frame_id = m_node_config.world_frame;
        msg.header.stamp    = Utils::getTime(m_accum_last_time);
        m_accum_cloud_pub->publish(msg);
    }

private:
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr m_lidar_sub;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_lidar_pc2_sub;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr m_imu_sub;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_body_cloud_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_world_cloud_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_accum_cloud_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr m_path_pub;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr m_odom_pub;

    rclcpp::TimerBase::SharedPtr m_timer;
    rclcpp::TimerBase::SharedPtr m_accum_timer;

    CloudType::Ptr            m_accum_cloud;
    double                    m_accum_last_time = 0.0;
    pcl::VoxelGrid<PointType> m_accum_voxel;

    StateData m_state_data;
    SyncPackage m_package;
    NodeConfig m_node_config;
    Config m_builder_config;
    std::shared_ptr<IESKF> m_kf;
    std::shared_ptr<MapBuilder> m_builder;
    // Guards m_builder (ikd-tree insertions in the timer vs. map extraction in
    // the save_map service, which runs on another executor thread).
    std::mutex m_builder_mutex;
    std::shared_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;
    rclcpp::CallbackGroup::SharedPtr m_service_cb_group;
    rclcpp::Service<interface::srv::SaveMaps>::SharedPtr m_save_map_srv;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    int ret = 0;
    try
    {
        auto node = std::make_shared<LIONode>();
        // Two threads: sensor callbacks + odometry timer stay in the default
        // (mutually exclusive) group; the save_map service runs on the other
        // thread so a slow PCD write cannot stall the odometry pipeline.
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);
        executor.spin();
    }
    catch (const std::exception &e)
    {
        RCLCPP_FATAL(rclcpp::get_logger("lio_node"), "%s", e.what());
        ret = 1;
    }
    rclcpp::shutdown();
    return ret;
}