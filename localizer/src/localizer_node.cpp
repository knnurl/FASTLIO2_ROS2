#include <queue>
#include <mutex>
#include <filesystem>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include "localizers/commons.h"
#include "localizers/icp_localizer.h"
#include "interface/srv/relocalize.hpp"
#include "interface/srv/is_valid.hpp"
#include "interface/srv/global_relocalize.hpp"
#include "interface/srv/start_mapping.hpp"
#include "interface/srv/stop_mapping.hpp"
#include <yaml-cpp/yaml.h>

using namespace std::chrono_literals;

struct NodeConfig
{
    std::string cloud_topic = "/fastlio2/body_cloud";
    std::string odom_topic = "/fastlio2/lio_odom";
    std::string map_frame = "map";
    std::string local_frame = "lidar";
    double update_hz = 1.0;
};

struct NodeState
{
    std::mutex message_mutex;
    std::mutex service_mutex;

    bool message_received = false;
    bool service_received = false;
    bool localize_success = false;
    rclcpp::Time last_send_tf_time = rclcpp::Clock().now();
    builtin_interfaces::msg::Time last_message_time;
    CloudType::Ptr last_cloud = std::make_shared<CloudType>();
    M3D last_r;                          // localmap_body_r
    V3D last_t;                          // localmap_body_t
    M3D last_offset_r = M3D::Identity(); // map_localmap_r
    V3D last_offset_t = V3D::Zero();     // map_localmap_t
    M4F initial_guess = M4F::Identity();

    // Auxiliary mapping state
    std::mutex mapping_mutex;
    bool mapping_active = false;
    float mapping_voxel_size = 0.05f;
    std::string mapping_output_path;
    CloudType::Ptr mapping_cloud = std::make_shared<CloudType>();
};

class LocalizerNode : public rclcpp::Node
{
public:
    LocalizerNode() : Node("localizer_node")
    {
        RCLCPP_INFO(this->get_logger(), "Localizer Node Started");
        loadParameters();
        rclcpp::QoS qos = rclcpp::QoS(10);
        m_cloud_sub.subscribe(this, m_config.cloud_topic, qos.get_rmw_qos_profile());
        m_odom_sub.subscribe(this, m_config.odom_topic, qos.get_rmw_qos_profile());

        m_tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

        m_sync = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>>>(message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>(10), m_cloud_sub, m_odom_sub);
        m_sync->setAgePenalty(0.1);
        m_sync->registerCallback(std::bind(&LocalizerNode::syncCB, this, std::placeholders::_1, std::placeholders::_2));
        m_localizer = std::make_shared<ICPLocalizer>(m_localizer_config);

        // Load startup map now that m_localizer is ready
        if (!m_startup_map_path.empty())
        {
            if (m_localizer->loadMap(m_startup_map_path))
                RCLCPP_INFO(this->get_logger(), "Map loaded at startup: %s", m_startup_map_path.c_str());
            else
                RCLCPP_ERROR(this->get_logger(), "Failed to load map at startup: %s", m_startup_map_path.c_str());
        }

        // ICP mode (original logic, unchanged)
        m_reloc_srv = this->create_service<interface::srv::Relocalize>(
            "relocalize",
            std::bind(&LocalizerNode::relocCB, this, std::placeholders::_1, std::placeholders::_2));

        // Renamed from relocalize_check → is_valid to match GUI and README
        m_reloc_check_srv = this->create_service<interface::srv::IsValid>(
            "is_valid",
            std::bind(&LocalizerNode::relocCheckCB, this, std::placeholders::_1, std::placeholders::_2));

        // Global relocalization (NDT+ICP, no pose hint required)
        m_global_reloc_srv = this->create_service<interface::srv::GlobalRelocalize>(
            "global_relocalize",
            std::bind(&LocalizerNode::globalRelocCB, this, std::placeholders::_1, std::placeholders::_2));

        // Auxiliary map update services
        m_start_mapping_srv = this->create_service<interface::srv::StartMapping>(
            "start_mapping",
            std::bind(&LocalizerNode::startMappingCB, this, std::placeholders::_1, std::placeholders::_2));

        m_stop_mapping_srv = this->create_service<interface::srv::StopMapping>(
            "stop_mapping",
            std::bind(&LocalizerNode::stopMappingCB, this, std::placeholders::_1, std::placeholders::_2));

        m_map_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("map_cloud", 10);

        m_timer = this->create_wall_timer(10ms, std::bind(&LocalizerNode::timerCB, this));
    }

    void loadParameters()
    {
        this->declare_parameter("config_path", "");
        this->declare_parameter("map_path", "");
        this->declare_parameter("cloud_topic", "");

        std::string config_path;
        this->get_parameter<std::string>("config_path", config_path);
        YAML::Node config = YAML::LoadFile(config_path);
        if (!config)
        {
            RCLCPP_WARN(this->get_logger(), "FAIL TO LOAD YAML FILE!");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "LOAD FROM YAML CONFIG PATH: %s", config_path.c_str());

        m_config.cloud_topic = config["cloud_topic"].as<std::string>();
        m_config.odom_topic = config["odom_topic"].as<std::string>();
        m_config.map_frame = config["map_frame"].as<std::string>();
        m_config.local_frame = config["local_frame"].as<std::string>();
        m_config.update_hz = config["update_hz"].as<double>();

        // ROS parameter overrides YAML (used by launch files)
        std::string cloud_topic_override;
        this->get_parameter<std::string>("cloud_topic", cloud_topic_override);
        if (!cloud_topic_override.empty())
            m_config.cloud_topic = cloud_topic_override;

        m_localizer_config.rough_scan_resolution = config["rough_scan_resolution"].as<double>();
        m_localizer_config.rough_map_resolution = config["rough_map_resolution"].as<double>();
        m_localizer_config.rough_max_iteration = config["rough_max_iteration"].as<int>();
        m_localizer_config.rough_score_thresh = config["rough_score_thresh"].as<double>();

        m_localizer_config.refine_scan_resolution = config["refine_scan_resolution"].as<double>();
        m_localizer_config.refine_map_resolution = config["refine_map_resolution"].as<double>();
        m_localizer_config.refine_max_iteration = config["refine_max_iteration"].as<int>();
        m_localizer_config.refine_score_thresh = config["refine_score_thresh"].as<double>();

        // Store map_path for deferred loading after m_localizer is constructed
        this->get_parameter<std::string>("map_path", m_startup_map_path);
    }

    void timerCB()
    {
        if (!m_state.message_received)
            return;

        rclcpp::Duration diff = rclcpp::Clock().now() - m_state.last_send_tf_time;

        bool update_tf = diff.seconds() > (1.0 / m_config.update_hz) && m_state.message_received;

        if (!update_tf)
        {
            sendBroadCastTF(m_state.last_message_time);
            return;
        }

        m_state.last_send_tf_time = rclcpp::Clock().now();

        M4F initial_guess = M4F::Identity();
        if (m_state.service_received)
        {
            std::lock_guard<std::mutex> lock(m_state.service_mutex);
            initial_guess = m_state.initial_guess;
        }
        else
        {
            std::lock_guard<std::mutex> lock(m_state.message_mutex);
            initial_guess.block<3, 3>(0, 0) = (m_state.last_offset_r * m_state.last_r).cast<float>();
            initial_guess.block<3, 1>(0, 3) = (m_state.last_offset_r * m_state.last_t + m_state.last_offset_t).cast<float>();
        }

        M3D current_local_r;
        V3D current_local_t;
        builtin_interfaces::msg::Time current_time;
        {
            std::lock_guard<std::mutex> lock(m_state.message_mutex);
            current_local_r = m_state.last_r;
            current_local_t = m_state.last_t;
            current_time = m_state.last_message_time;
            m_localizer->setInput(m_state.last_cloud);
        }

        bool result = m_localizer->align(initial_guess);
        if (result)
        {
            M3D map_body_r = initial_guess.block<3, 3>(0, 0).cast<double>();
            V3D map_body_t = initial_guess.block<3, 1>(0, 3).cast<double>();
            m_state.last_offset_r = map_body_r * current_local_r.transpose();
            m_state.last_offset_t = -map_body_r * current_local_r.transpose() * current_local_t + map_body_t;
            if (!m_state.localize_success && m_state.service_received)
            {
                std::lock_guard<std::mutex> lock(m_state.service_mutex);
                m_state.localize_success = true;
                m_state.service_received = false;
            }
        }
        sendBroadCastTF(current_time);
        publishMapCloud(current_time);
    }

    void syncCB(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud_msg, const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg)
    {
        {
            std::lock_guard<std::mutex> lock(m_state.message_mutex);

            pcl::fromROSMsg(*cloud_msg, *m_state.last_cloud);

            m_state.last_r = Eigen::Quaterniond(odom_msg->pose.pose.orientation.w,
                                                odom_msg->pose.pose.orientation.x,
                                                odom_msg->pose.pose.orientation.y,
                                                odom_msg->pose.pose.orientation.z)
                                 .toRotationMatrix();
            m_state.last_t = V3D(odom_msg->pose.pose.position.x,
                                 odom_msg->pose.pose.position.y,
                                 odom_msg->pose.pose.position.z);
            m_state.last_message_time = cloud_msg->header.stamp;
            if (!m_state.message_received)
            {
                m_state.message_received = true;
                m_config.local_frame = odom_msg->header.frame_id;
            }
        }

        // Accumulate cloud for auxiliary mapping
        {
            std::lock_guard<std::mutex> lock(m_state.mapping_mutex);
            if (m_state.mapping_active)
            {
                CloudType scan;
                pcl::fromROSMsg(*cloud_msg, scan);
                *m_state.mapping_cloud += scan;
            }
        }
    }

    void sendBroadCastTF(builtin_interfaces::msg::Time &time)
    {
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.frame_id = m_config.map_frame;
        transformStamped.child_frame_id = m_config.local_frame;
        transformStamped.header.stamp = time;
        Eigen::Quaterniond q(m_state.last_offset_r);
        V3D t = m_state.last_offset_t;
        transformStamped.transform.translation.x = t.x();
        transformStamped.transform.translation.y = t.y();
        transformStamped.transform.translation.z = t.z();
        transformStamped.transform.rotation.x = q.x();
        transformStamped.transform.rotation.y = q.y();
        transformStamped.transform.rotation.z = q.z();
        transformStamped.transform.rotation.w = q.w();
        m_tf_broadcaster->sendTransform(transformStamped);
    }

    // ICP mode (original logic, unchanged)
    void relocCB(const std::shared_ptr<interface::srv::Relocalize::Request> request, std::shared_ptr<interface::srv::Relocalize::Response> response)
    {
        std::string pcd_path = request->pcd_path;
        float x = request->x;
        float y = request->y;
        float z = request->z;
        float yaw = request->yaw;
        float roll = request->roll;
        float pitch = request->pitch;

        if (!std::filesystem::exists(pcd_path))
        {
            response->success = false;
            response->message = "pcd file not found";
            return;
        }

        Eigen::AngleAxisd yaw_angle = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd roll_angle = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch_angle = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY());
        bool load_flag = m_localizer->loadMap(pcd_path);
        if (!load_flag)
        {
            response->success = false;
            response->message = "load map failed";
            return;
        }
        {
            std::lock_guard<std::mutex> lock(m_state.message_mutex);
            m_state.initial_guess.setIdentity();
            m_state.initial_guess.block<3, 3>(0, 0) = (yaw_angle * roll_angle * pitch_angle).toRotationMatrix().cast<float>();
            m_state.initial_guess.block<3, 1>(0, 3) = V3F(x, y, z);
            m_state.service_received = true;
            m_state.localize_success = false;
        }

        response->success = true;
        response->message = "relocalize success";
        return;
    }

    // Global relocalization — no pose hint, uses currently loaded map
    void globalRelocCB(
        const std::shared_ptr<interface::srv::GlobalRelocalize::Request> request,
        std::shared_ptr<interface::srv::GlobalRelocalize::Response> response)
    {
        // Optionally reload the map from a new path
        if (!request->pcd_path.empty())
        {
            if (!std::filesystem::exists(request->pcd_path))
            {
                response->success = false;
                response->message = "pcd file not found: " + request->pcd_path;
                return;
            }
            if (!m_localizer->loadMap(request->pcd_path))
            {
                response->success = false;
                response->message = "Failed to load map: " + request->pcd_path;
                return;
            }
        }

        // Force re-localization from identity (global search)
        {
            std::lock_guard<std::mutex> lock(m_state.service_mutex);
            m_state.initial_guess.setIdentity();
            m_state.service_received = true;
            m_state.localize_success = false;
        }

        // Run one alignment synchronously to return the fitness score
        M4F guess = M4F::Identity();
        {
            std::lock_guard<std::mutex> lock(m_state.message_mutex);
            RCLCPP_INFO(this->get_logger(),
                "global_relocalize: scan_pts=%zu map_rough=%zu map_refine=%zu",
                m_state.last_cloud->size(),
                m_localizer->roughMap()->size(),
                m_localizer->refineMap()->size());
            m_localizer->setInput(m_state.last_cloud);
        }
        bool ok = m_localizer->align(guess);
        if (!ok)
        {
            response->success = false;
            response->message = "ICP alignment failed";
            return;
        }

        // Extract pose from alignment result
        M3D r = guess.block<3, 3>(0, 0).cast<double>();
        V3D t = guess.block<3, 1>(0, 3).cast<double>();
        Eigen::Quaterniond q(r);
        double yaw_rad = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
                                    1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));

        // Update the offset so the TF broadcaster uses this result immediately
        {
            std::lock_guard<std::mutex> lock(m_state.service_mutex);
            m_state.last_offset_r = r * m_state.last_r.transpose();
            m_state.last_offset_t = -r * m_state.last_r.transpose() * m_state.last_t + t;
            m_state.localize_success = true;
            m_state.service_received = false;
        }

        response->success      = true;
        response->message      = "Global relocalization succeeded";
        response->fitness_score = m_localizer->getScore();
        response->x            = static_cast<float>(t.x());
        response->y            = static_cast<float>(t.y());
        response->z            = static_cast<float>(t.z());
        response->yaw_deg      = static_cast<float>(yaw_rad * 180.0 / M_PI);
    }

    void relocCheckCB(const std::shared_ptr<interface::srv::IsValid::Request> request, std::shared_ptr<interface::srv::IsValid::Response> response)
    {
        std::lock_guard<std::mutex> lock(m_state.service_mutex);
        if (request->code == 1)
            response->valid = true;
        else
            response->valid = m_state.localize_success;
        return;
    }

    void startMappingCB(
        const std::shared_ptr<interface::srv::StartMapping::Request> request,
        std::shared_ptr<interface::srv::StartMapping::Response> response)
    {
        std::lock_guard<std::mutex> lock(m_state.mapping_mutex);
        if (m_state.mapping_active)
        {
            response->success = false;
            response->message = "Map update already in progress";
            return;
        }
        m_state.mapping_active = true;
        m_state.mapping_voxel_size = request->voxel_size > 0.0f ? request->voxel_size : 0.05f;
        m_state.mapping_output_path = request->output_path;
        m_state.mapping_cloud->clear();
        response->success = true;
        response->message = "Map update started";
        RCLCPP_INFO(this->get_logger(), "Auxiliary map update started (voxel=%.3f)", m_state.mapping_voxel_size);
    }

    void stopMappingCB(
        const std::shared_ptr<interface::srv::StopMapping::Request> request,
        std::shared_ptr<interface::srv::StopMapping::Response> response)
    {
        CloudType::Ptr accumulated;
        float voxel_size;
        std::string out_path;
        {
            std::lock_guard<std::mutex> lock(m_state.mapping_mutex);
            if (!m_state.mapping_active)
            {
                response->success = false;
                response->message = "No map update in progress";
                return;
            }
            m_state.mapping_active = false;
            accumulated = m_state.mapping_cloud;
            voxel_size = m_state.mapping_voxel_size;
            out_path = request->output_path.empty() ? m_state.mapping_output_path : request->output_path;
            m_state.mapping_cloud = std::make_shared<CloudType>();
        }

        if (accumulated->empty())
        {
            response->success = false;
            response->message = "No scan data accumulated";
            return;
        }

        // Voxel-filter the accumulated cloud
        pcl::VoxelGrid<PointType> vf;
        vf.setInputCloud(accumulated);
        vf.setLeafSize(voxel_size, voxel_size, voxel_size);
        CloudType::Ptr filtered(new CloudType);
        vf.filter(*filtered);

        if (!out_path.empty())
        {
            auto parent = std::filesystem::path(out_path).parent_path();
            if (!parent.empty()) {
                std::error_code ec;
                std::filesystem::create_directories(parent, ec);
            }
            if (pcl::io::savePCDFileBinary(out_path, *filtered) != 0)
            {
                response->success = false;
                response->message = "Failed to save PCD to: " + out_path;
                return;
            }
            // Hot-reload: update the ICP localizer with the new merged map
            m_localizer->loadMap(out_path);
            RCLCPP_INFO(this->get_logger(), "Map update saved and reloaded: %s (%zu pts)",
                        out_path.c_str(), filtered->size());
        }

        response->success = true;
        response->message = "Map update complete (" + std::to_string(filtered->size()) + " points)";
    }

    void publishMapCloud(builtin_interfaces::msg::Time &time)
    {
        if (m_map_cloud_pub->get_subscription_count() < 1)
            return;
        CloudType::Ptr map_cloud = m_localizer->refineMap();
        if (map_cloud->size() < 1)
            return;
        sensor_msgs::msg::PointCloud2 map_cloud_msg;
        pcl::toROSMsg(*map_cloud, map_cloud_msg);
        map_cloud_msg.header.frame_id = m_config.map_frame;
        map_cloud_msg.header.stamp = time;
        m_map_cloud_pub->publish(map_cloud_msg);
    }

private:
    NodeConfig m_config;
    NodeState m_state;

    ICPConfig m_localizer_config;
    std::string m_startup_map_path;
    std::shared_ptr<ICPLocalizer> m_localizer;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> m_cloud_sub;
    message_filters::Subscriber<nav_msgs::msg::Odometry> m_odom_sub;
    rclcpp::TimerBase::SharedPtr m_timer;
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>>> m_sync;
    std::shared_ptr<tf2_ros::TransformBroadcaster> m_tf_broadcaster;
    rclcpp::Service<interface::srv::Relocalize>::SharedPtr m_reloc_srv;
    rclcpp::Service<interface::srv::IsValid>::SharedPtr m_reloc_check_srv;
    rclcpp::Service<interface::srv::GlobalRelocalize>::SharedPtr m_global_reloc_srv;
    rclcpp::Service<interface::srv::StartMapping>::SharedPtr m_start_mapping_srv;
    rclcpp::Service<interface::srv::StopMapping>::SharedPtr m_stop_mapping_srv;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_map_cloud_pub;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LocalizerNode>());
    rclcpp::shutdown();
    return 0;
}
