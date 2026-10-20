#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include "hba/hba.h"
#include <pcl/filters/voxel_grid.h>
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <chrono>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include "interface/srv/refine_map.hpp"
#include "interface/srv/save_poses.hpp"

// Closed-loop integration: HBATrigger / HBACompletion from health monitor
#include "fastlio2_health/msg/hba_trigger.hpp"
#include "fastlio2_health/msg/hba_completion.hpp"

using namespace std::chrono_literals;
void fromStr(const std::string &str, std::string &file_name, Pose &pose)
{
    std::stringstream ss(str);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(ss, token, ' '))
    {
        tokens.push_back(token);
    }
    assert(tokens.size() == 8);
    file_name = tokens[0];
    pose.t = V3D(std::stod(tokens[1]), std::stod(tokens[2]), std::stod(tokens[3]));
    pose.r = Eigen::Quaterniond(std::stod(tokens[4]), std::stod(tokens[5]), std::stod(tokens[6]), std::stod(tokens[7])).normalized().toRotationMatrix();
}

struct NodeConfig
{
    double scan_resolution = 0.1;
};

class HBANode : public rclcpp::Node
{
public:
    HBANode() : Node("hba_node")
    {
        RCLCPP_INFO(this->get_logger(), "HBA node started");
        loadParameters();
        m_hba = std::make_shared<HBA>(m_hba_config);
        m_voxel_grid.setLeafSize(m_node_config.scan_resolution, m_node_config.scan_resolution, m_node_config.scan_resolution);
        m_refine_map_srv = this->create_service<interface::srv::RefineMap>(
            "refine_map",
            std::bind(&HBANode::refineMapCB, this, std::placeholders::_1, std::placeholders::_2));
        m_save_poses_srv = this->create_service<interface::srv::SavePoses>(
            "save_poses",
            std::bind(&HBANode::savePosesCB, this, std::placeholders::_1, std::placeholders::_2));
        m_timer = this->create_wall_timer(100ms, std::bind(&HBANode::mainCB, this));
        m_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("map_points", 10);

        // ── Closed-loop integration ──
        m_hba_completion_pub = this->create_publisher<fastlio2_health::msg::HBACompletion>(
            "/hba/completion", 10);

        m_hba_trigger_sub = this->create_subscription<fastlio2_health::msg::HBATrigger>(
            "/hba/trigger", 10,
            std::bind(&HBANode::onHbaTrigger, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Subscribed to /hba/trigger, publishing on /hba/completion");
    }

    void loadParameters()
    {
        this->declare_parameter("config_path", "");
        this->declare_parameter("maps_path", "");
        std::string config_path;
        this->get_parameter<std::string>("config_path", config_path);
        this->get_parameter<std::string>("maps_path", m_default_maps_path);

        YAML::Node config = YAML::LoadFile(config_path);
        if (!config)
        {
            RCLCPP_WARN(this->get_logger(), "FAIL TO LOAD YAML FILE!");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "LOAD FROM YAML CONFIG PATH: %s", config_path.c_str());

        m_node_config.scan_resolution = config["scan_resolution"].as<double>();
        m_hba_config.window_size = config["window_size"].as<int>();
        m_hba_config.stride = config["stride"].as<int>();
        m_hba_config.voxel_size = config["voxel_size"].as<double>();
        m_hba_config.min_point_num = config["min_point_num"].as<int>();
        m_hba_config.max_layer = config["max_layer"].as<int>();
        m_hba_config.plane_thresh = config["plane_thresh"].as<double>();
        m_hba_config.ba_max_iter = config["ba_max_iter"].as<size_t>();
        m_hba_config.hba_iter = config["hba_iter"].as<size_t>();
        m_hba_config.down_sample = config["down_sample"].as<double>();
    }

    bool loadFromPath(const std::string &maps_path)
    {
        if (!std::filesystem::exists(maps_path))
            return false;

        std::filesystem::path p_dir(maps_path);
        std::filesystem::path pcd_dir = p_dir / "patches";
        std::filesystem::path txt_file = p_dir / "poses.txt";

        if (!std::filesystem::exists(pcd_dir) || !std::filesystem::exists(txt_file))
            return false;

        // Reset HBA state for fresh load
        m_hba = std::make_shared<HBA>(m_hba_config);

        std::ifstream ifs(txt_file);
        std::string line;
        std::string file_name;
        Pose pose;
        pcl::PCDReader reader;
        while (std::getline(ifs, line))
        {
            fromStr(line, file_name, pose);
            std::filesystem::path pcd_file = p_dir / "patches" / file_name;
            if (!std::filesystem::exists(pcd_file))
            {
                std::cerr << "pcd file not found" << std::endl;
                continue;
            }
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
            reader.read(pcd_file, *cloud);
            m_voxel_grid.setInputCloud(cloud);
            m_voxel_grid.filter(*cloud);
            m_hba->insert(cloud, pose);
        }
        RCLCPP_INFO(this->get_logger(), "Loaded %lu poses from %s", m_hba->poses().size(), maps_path.c_str());
        return m_hba->poses().size() > 0;
    }

    void refineMapCB(const std::shared_ptr<interface::srv::RefineMap::Request> request, std::shared_ptr<interface::srv::RefineMap::Response> response)
    {
        std::lock_guard<std::mutex> lock(m_service_mutex);
        if (!loadFromPath(request->maps_path))
        {
            response->success = false;
            response->message = "Failed to load from " + request->maps_path;
            return;
        }
        response->success = true;
        response->message = "load poses success!";
        m_do_optimize = true;
        return;
    }

    void savePosesCB(const std::shared_ptr<interface::srv::SavePoses::Request> request, std::shared_ptr<interface::srv::SavePoses::Response> response)
    {
        std::lock_guard<std::mutex> lock(m_service_mutex);
        std::filesystem::path file_path(request->file_path);
        std::filesystem::path par_path = file_path.parent_path();
        if (!std::filesystem::exists(par_path))
        {
            response->success = false;
            response->message = "parent path not exists";
            return;
        }
        if (m_hba->poses().size() < 1)
        {
            response->success = false;
            response->message = "poses is empty";
            return;
        }

        if (std::filesystem::exists(file_path))
        {
            std::filesystem::remove(file_path);
        }

        m_hba->writePoses(file_path);
        response->success = true;
        response->message = "save poses success!";
        return;
    }

    void onHbaTrigger(const fastlio2_health::msg::HBATrigger::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(m_service_mutex);

        RCLCPP_INFO(this->get_logger(), "HBA trigger received: reason=%s, nodes=%zu, fitness=%.3f",
                     msg->trigger_reason.c_str(), msg->node_ids.size(), msg->trigger_fitness_score);

        // If no data loaded, try loading from default maps_path
        if (m_hba->poses().empty() && !m_default_maps_path.empty()) {
            RCLCPP_INFO(this->get_logger(), "No data loaded, attempting load from: %s",
                        m_default_maps_path.c_str());
            if (!loadFromPath(m_default_maps_path)) {
                RCLCPP_ERROR(this->get_logger(), "Failed to load data for HBA trigger");
                publishCompletion(false, msg->node_ids, 0.0f);
                return;
            }
        }

        if (m_hba->poses().empty()) {
            RCLCPP_ERROR(this->get_logger(), "No poses available for optimization");
            publishCompletion(false, msg->node_ids, 0.0f);
            return;
        }

        // Store pre-optimization poses for correction magnitude computation
        m_pre_opt_poses = m_hba->poses();
        m_pending_trigger_node_ids = msg->node_ids;
        m_do_optimize = true;
        m_publish_completion_on_done = true;
    }

    void mainCB()
    {
        {
            std::lock_guard<std::mutex> lock(m_service_mutex);
            if (!m_do_optimize)
                return;
            RCLCPP_WARN(this->get_logger(), "START OPTIMIZE");
            publishMap();
            for (size_t i = 0; i < m_hba_config.hba_iter; i++)
            {
                RCLCPP_INFO(this->get_logger(), "======HBA ITER %lu START======", i + 1);
                m_hba->optimize();
                publishMap();
                RCLCPP_INFO(this->get_logger(), "======HBA ITER %lu END========", i + 1);
            }

            m_do_optimize = false;
            RCLCPP_WARN(this->get_logger(), "END OPTIMIZE");

            // Publish completion for trigger-based integration
            if (m_publish_completion_on_done) {
                float mean_correction = computeMeanCorrection();
                publishCompletion(true, m_pending_trigger_node_ids, mean_correction);
                m_publish_completion_on_done = false;
            }
        }
    }

    void publishMap()
    {
        if (m_cloud_pub->get_subscription_count() < 1)
            return;
        if (m_hba->poses().size() < 1)
            return;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = m_hba->getMapPoints();
        m_voxel_grid.setInputCloud(cloud);
        m_voxel_grid.filter(*cloud);
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.frame_id = "map";
        cloud_msg.header.stamp = this->now();
        m_cloud_pub->publish(cloud_msg);
    }

private:
    float computeMeanCorrection()
    {
        const auto &post_poses = m_hba->poses();
        if (m_pre_opt_poses.size() != post_poses.size() || post_poses.empty())
            return 0.0f;

        double total_displacement = 0.0;
        for (size_t i = 0; i < post_poses.size(); i++) {
            total_displacement += (post_poses[i].t - m_pre_opt_poses[i].t).norm();
        }
        return static_cast<float>(total_displacement / post_poses.size());
    }

    void publishCompletion(bool success, const std::vector<int> &node_ids, float mean_correction)
    {
        auto comp = fastlio2_health::msg::HBACompletion();
        comp.header.stamp = this->now();
        comp.header.frame_id = "map";
        comp.success = success;
        comp.node_ids = node_ids;
        comp.mean_correction_magnitude = mean_correction;

        // Attach the corrected map cloud
        if (success && m_hba->poses().size() > 0) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud = m_hba->getMapPoints();
            m_voxel_grid.setInputCloud(map_cloud);
            m_voxel_grid.filter(*map_cloud);
            pcl::toROSMsg(*map_cloud, comp.corrected_pcd);
            comp.corrected_pcd.header.frame_id = "map";
            comp.corrected_pcd.header.stamp = this->now();

            // Attach corrected transforms (one per pose)
            for (size_t i = 0; i < m_hba->poses().size(); i++) {
                geometry_msgs::msg::TransformStamped tf;
                tf.header.stamp = this->now();
                tf.header.frame_id = "map";
                tf.child_frame_id = "hba_pose_" + std::to_string(i);
                const Pose &p = m_hba->poses()[i];
                tf.transform.translation.x = p.t.x();
                tf.transform.translation.y = p.t.y();
                tf.transform.translation.z = p.t.z();
                Eigen::Quaterniond q(p.r);
                tf.transform.rotation.x = q.x();
                tf.transform.rotation.y = q.y();
                tf.transform.rotation.z = q.z();
                tf.transform.rotation.w = q.w();
                comp.corrected_transforms.push_back(tf);
            }
        }

        m_hba_completion_pub->publish(comp);
        RCLCPP_INFO(this->get_logger(), "Published HBA completion: success=%s, correction=%.4fm",
                     success ? "true" : "false", mean_correction);
    }

    NodeConfig m_node_config;
    HBAConfig m_hba_config;
    std::shared_ptr<HBA> m_hba;
    rclcpp::Service<interface::srv::RefineMap>::SharedPtr m_refine_map_srv;
    rclcpp::Service<interface::srv::SavePoses>::SharedPtr m_save_poses_srv;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr m_cloud_pub;

    pcl::VoxelGrid<pcl::PointXYZI> m_voxel_grid;
    std::mutex m_service_mutex;
    bool m_do_optimize = false;
    rclcpp::TimerBase::SharedPtr m_timer;

    // Closed-loop integration
    rclcpp::Publisher<fastlio2_health::msg::HBACompletion>::SharedPtr m_hba_completion_pub;
    rclcpp::Subscription<fastlio2_health::msg::HBATrigger>::SharedPtr m_hba_trigger_sub;
    std::string m_default_maps_path;
    Vec<Pose> m_pre_opt_poses;
    std::vector<int> m_pending_trigger_node_ids;
    bool m_publish_completion_on_done = false;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HBANode>());
    rclcpp::shutdown();
    return 0;
}
