#include <queue>
#include <mutex>
#include <atomic>
#include <future>
#include <fstream>
#include <filesystem>
#include <cstdlib>
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
#include <std_msgs/msg/string.hpp>

#include "localizers/commons.h"
#include "localizers/icp_localizer.h"
#include "localizers/global_localizer.h"
#include "localizer_fsm.h"
#include "interface/srv/relocalize.hpp"
#include "interface/srv/is_valid.hpp"
#include "interface/srv/global_relocalize.hpp"
#include "interface/srv/start_mapping.hpp"
#include "interface/srv/stop_mapping.hpp"
#include <pcl/common/transforms.h>
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

enum class LocalizerMode
{
    ICP,    // Requires manual service call with initial pose
    GLOBAL  // Auto-localizes on startup without pose hint
};

enum class LocalizerState
{
    IDLE,       // Not yet received any sensor data
    SEARCHING,  // Running global search
    TRACKING,   // ICP tracking active
    LOST        // ICP fitness too high, needs re-localization
};

struct NodeConfig
{
    std::string cloud_topic       = "/fastlio2/body_cloud";
    std::string world_cloud_topic = "/fastlio2/world_cloud";
    std::string odom_topic        = "/fastlio2/lio_odom";
    std::string map_frame   = "map";
    std::string local_frame = "lidar";
    double update_hz        = 1.0;

    // Global mode
    LocalizerMode mode       = LocalizerMode::ICP;
    std::string map_path     = "";
    std::string pose_save_path = "/tmp/fastlio2_last_pose.yaml"; // persisted between runs
    double lost_thresh       = 1.5;
    double lost_patience_sec = 3.0;
    double pose_save_hz      = 0.5; // save pose to disk at this rate when tracking
    int min_cloud_points     = 50;  // below this, skip ICP (degenerate scan from lio)
    // ICP jump guard: if the ICP result moves more than this from the predicted
    // pose it is likely a wrong local minimum and is rejected as a failure.
    double icp_max_jump_m    = 1.0;
};

struct NodeState
{
    // Single mutex for all fields below. They are shared between the sensor/
    // timer callbacks (default group) and the heavy services (relocalize,
    // global_relocalize, stop_mapping), which run on another executor thread.
    std::mutex mutex;

    bool message_received   = false;
    bool service_received   = false;
    bool localize_success   = false;

    // Set from the node clock in the constructor: all timing must use the
    // node clock (sim-time aware), and mixing clock types in rclcpp::Time
    // arithmetic throws.
    rclcpp::Time last_send_tf_time;
    rclcpp::Time last_pose_save_time;
    builtin_interfaces::msg::Time last_message_time;

    CloudType::Ptr last_cloud = std::make_shared<CloudType>();
    M3D last_r;
    V3D last_t;
    M3D last_offset_r = M3D::Identity();
    V3D last_offset_t = V3D::Zero();
    M4F initial_guess = M4F::Identity();

    // Global mode
    LocalizerState localizer_state  = LocalizerState::IDLE;
    int consecutive_icp_failures = 0;
};

class LocalizerNode : public rclcpp::Node
{
public:
    LocalizerNode() : Node("localizer_node")
    {
        RCLCPP_INFO(this->get_logger(), "Localizer Node Started");
        loadParameters();

        m_state.last_send_tf_time   = this->get_clock()->now();
        m_state.last_pose_save_time = this->get_clock()->now();

        rclcpp::QoS qos = rclcpp::QoS(10);
        m_cloud_sub.subscribe(this, m_config.cloud_topic, qos.get_rmw_qos_profile());
        m_odom_sub.subscribe(this, m_config.odom_topic, qos.get_rmw_qos_profile());

        m_tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

        using SyncPolicy = message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>;
        m_sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), m_cloud_sub, m_odom_sub);
        m_sync->setAgePenalty(0.1);
        m_sync->registerCallback(
            std::bind(&LocalizerNode::syncCB, this, std::placeholders::_1, std::placeholders::_2));

        m_icp_localizer    = std::make_shared<ICPLocalizer>(m_icp_config);
        m_global_localizer = std::make_shared<GlobalLocalizer>(m_global_config);

        // Heavy services (PCD loads, global search, map merge+save) run in
        // their own callback group so the MultiThreadedExecutor keeps the
        // localization timer — and with it the map→local TF — alive while
        // they work. Light services stay in the default group.
        m_heavy_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        m_reloc_srv = this->create_service<interface::srv::Relocalize>(
            "relocalize",
            std::bind(&LocalizerNode::relocCB, this, std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, m_heavy_cb_group);

        m_reloc_check_srv = this->create_service<interface::srv::IsValid>(
            "relocalize_check",
            std::bind(&LocalizerNode::relocCheckCB, this, std::placeholders::_1, std::placeholders::_2));

        m_global_reloc_srv = this->create_service<interface::srv::GlobalRelocalize>(
            "global_relocalize",
            std::bind(&LocalizerNode::globalRelocCB, this, std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, m_heavy_cb_group);

        m_start_mapping_srv = this->create_service<interface::srv::StartMapping>(
            "start_mapping",
            std::bind(&LocalizerNode::startMappingCB, this, std::placeholders::_1, std::placeholders::_2));

        m_stop_mapping_srv = this->create_service<interface::srv::StopMapping>(
            "stop_mapping",
            std::bind(&LocalizerNode::stopMappingCB, this, std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default, m_heavy_cb_group);

        m_map_cloud_pub      = this->create_publisher<sensor_msgs::msg::PointCloud2>("map_cloud", 10);
        m_mapping_cloud_pub  = this->create_publisher<sensor_msgs::msg::PointCloud2>("mapping_cloud", 10);
        m_status_pub         = this->create_publisher<std_msgs::msg::String>("localizer_status", 10);

        // World cloud subscriber — used for map-update accumulation so the
        // full LIO-resolution scan (in lidar/world frame) is accumulated in
        // map frame, matching the quality of a pure-mapping session.
        m_world_cloud_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            m_config.world_cloud_topic,
            rclcpp::QoS(10),
            std::bind(&LocalizerNode::worldCloudCB, this, std::placeholders::_1));

        m_timer             = this->create_wall_timer(10ms,   std::bind(&LocalizerNode::timerCB, this));
        m_mapping_cloud_timer = this->create_wall_timer(1000ms, std::bind(&LocalizerNode::publishMappingCloud, this));
        m_status_timer      = this->create_wall_timer(1000ms, std::bind(&LocalizerNode::publishStatus, this));

        // Auto-load map in GLOBAL mode
        if (m_config.mode == LocalizerMode::GLOBAL && !m_config.map_path.empty())
        {
            if (!std::filesystem::exists(m_config.map_path))
            {
                RCLCPP_ERROR(this->get_logger(), "map_path does not exist: %s", m_config.map_path.c_str());
            }
            else
            {
                RCLCPP_INFO(this->get_logger(), "GLOBAL mode: Loading map from %s ...", m_config.map_path.c_str());
                bool ok = m_global_localizer->loadMap(m_config.map_path);
                bool icp_ok = m_icp_localizer->loadMap(m_config.map_path);
                if (ok && icp_ok)
                {
                    RCLCPP_INFO(this->get_logger(), "Map loaded. Waiting for sensor data to begin global search...");
                    m_state.localizer_state = LocalizerState::IDLE;
                }
                else
                {
                    RCLCPP_ERROR(this->get_logger(), "Failed to load map!");
                }
            }
        }
    }

    void loadParameters()
    {
        this->declare_parameter("config_path", "");
        this->declare_parameter("map_path", "");       // can override YAML value
        this->declare_parameter("cloud_topic", "");    // can override YAML value

        std::string config_path;
        this->get_parameter<std::string>("config_path", config_path);

        if (config_path.empty())
            throw std::runtime_error(
                "config_path parameter is not set (pass -p config_path:=<localizer.yaml>)");

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
        RCLCPP_INFO(this->get_logger(), "LOAD FROM YAML: %s", config_path.c_str());

        m_config.cloud_topic = yamlRequired<std::string>(config, "cloud_topic");
        m_config.odom_topic  = yamlRequired<std::string>(config, "odom_topic");

        // ROS2 parameter overrides YAML (used by launch files for people filter routing)
        std::string param_cloud_topic;
        this->get_parameter<std::string>("cloud_topic", param_cloud_topic);
        if (!param_cloud_topic.empty())
            m_config.cloud_topic = param_cloud_topic;
        m_config.map_frame   = yamlRequired<std::string>(config, "map_frame");
        m_config.local_frame = yamlRequired<std::string>(config, "local_frame");
        m_config.update_hz   = yamlRequired<double>(config, "update_hz");
        if (m_config.update_hz <= 0.0)
            throw std::runtime_error("update_hz must be > 0");

        // ICP config (existing)
        m_icp_config.rough_scan_resolution = yamlRequired<double>(config, "rough_scan_resolution");
        m_icp_config.rough_map_resolution  = yamlRequired<double>(config, "rough_map_resolution");
        m_icp_config.rough_max_iteration   = yamlRequired<int>(config, "rough_max_iteration");
        m_icp_config.rough_score_thresh    = yamlRequired<double>(config, "rough_score_thresh");
        if (config["rough_max_corr_dist"])
            m_icp_config.rough_max_corr_dist = config["rough_max_corr_dist"].as<double>();
        m_icp_config.refine_scan_resolution= yamlRequired<double>(config, "refine_scan_resolution");
        m_icp_config.refine_map_resolution = yamlRequired<double>(config, "refine_map_resolution");
        m_icp_config.refine_max_iteration  = yamlRequired<int>(config, "refine_max_iteration");
        m_icp_config.refine_score_thresh   = yamlRequired<double>(config, "refine_score_thresh");
        if (config["refine_max_corr_dist"])
            m_icp_config.refine_max_corr_dist = config["refine_max_corr_dist"].as<double>();

        if (config["rough_confidence_thresh"])
            m_icp_config.rough_confidence_thresh = config["rough_confidence_thresh"].as<double>();

        if (config["icp_recovery_corr_scale"])
            m_icp_config.recovery_corr_scale = config["icp_recovery_corr_scale"].as<double>();

        // Global localization config (new)
        if (config["localizer_mode"] && config["localizer_mode"].as<std::string>() == "global")
            m_config.mode = LocalizerMode::GLOBAL;

        if (config["map_path"])
            m_config.map_path = config["map_path"].as<std::string>();

        // ROS2 parameter overrides YAML (useful for launch file arguments)
        std::string param_map_path;
        this->get_parameter<std::string>("map_path", param_map_path);
        if (!param_map_path.empty())
            m_config.map_path = param_map_path;

        // Allow a home-relative map path ("~/fastlio2_ws/maps/map.pcd").
        if (m_config.map_path.rfind("~/", 0) == 0)
            if (const char *home = std::getenv("HOME"))
                m_config.map_path = std::string(home) + m_config.map_path.substr(1);

        if (config["lost_thresh"])
            m_config.lost_thresh = config["lost_thresh"].as<double>();

        if (config["lost_patience_sec"])
            m_config.lost_patience_sec = config["lost_patience_sec"].as<double>();

        if (config["global_xy_step_m"])
            m_global_config.xy_step_m = config["global_xy_step_m"].as<double>();

        if (config["global_yaw_step_deg"])
            m_global_config.yaw_step_deg = config["global_yaw_step_deg"].as<double>();

        if (config["global_ndt_resolution"])
            m_global_config.ndt_resolution = config["global_ndt_resolution"].as<double>();

        if (config["global_top_k"])
            m_global_config.top_k = config["global_top_k"].as<int>();

        if (config["global_icp_fitness_thresh"])
            m_global_config.icp_fitness_thresh = config["global_icp_fitness_thresh"].as<double>();

        if (config["global_ndt_score_thresh"])
            m_global_config.ndt_score_thresh = config["global_ndt_score_thresh"].as<double>();

        if (config["global_icp_max_corr_dist"])
            m_global_config.icp_max_corr_dist = config["global_icp_max_corr_dist"].as<double>();

        if (config["global_scan_voxel_coarse"])
            m_global_config.scan_voxel_coarse = config["global_scan_voxel_coarse"].as<double>();

        if (config["global_map_voxel"])
            m_global_config.map_voxel = config["global_map_voxel"].as<double>();

        if (config["global_prior_icp_max_corr_dist"])
            m_global_config.prior_icp_max_corr_dist = config["global_prior_icp_max_corr_dist"].as<double>();

        if (config["global_prior_icp_fitness_thresh"])
            m_global_config.prior_icp_fitness_thresh = config["global_prior_icp_fitness_thresh"].as<double>();

        if (config["pose_save_path"])
            m_config.pose_save_path = config["pose_save_path"].as<std::string>();

        if (config["pose_save_hz"])
            m_config.pose_save_hz = config["pose_save_hz"].as<double>();

        if (config["min_cloud_points"])
            m_config.min_cloud_points = config["min_cloud_points"].as<int>();

        if (config["icp_max_jump_m"])
            m_config.icp_max_jump_m = config["icp_max_jump_m"].as<double>();

        if (config["world_cloud_topic"])
            m_config.world_cloud_topic = config["world_cloud_topic"].as<std::string>();
    }

    // ─── Pose persistence helpers ──────────────────────────────────────────────
    bool savePose(const Eigen::Matrix4f &T)
    {
        // map_path can be swapped by stop_mapping on the service thread.
        std::string map_path;
        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            map_path = m_config.map_path;
        }
        try
        {
            YAML::Emitter out;
            out << YAML::BeginMap;
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    out << YAML::Key << ("m" + std::to_string(r) + std::to_string(c))
                        << YAML::Value << T(r, c);

            // Record the map file's modification time so loadPose() can detect
            // if the map has changed between sessions and reject the stale prior.
            if (!map_path.empty() && std::filesystem::exists(map_path))
            {
                auto mtime = std::filesystem::last_write_time(map_path);
                auto mtime_sec = std::chrono::duration_cast<std::chrono::seconds>(
                    mtime.time_since_epoch()).count();
                out << YAML::Key << "map_mtime" << YAML::Value << mtime_sec;
            }

            out << YAML::EndMap;

            std::ofstream fs(m_config.pose_save_path);
            if (!fs.is_open()) return false;
            fs << out.c_str();
            return true;
        }
        catch (...) { return false; }
    }

    bool loadPose(Eigen::Matrix4f &T)
    {
        if (!std::filesystem::exists(m_config.pose_save_path))
            return false;
        std::string map_path;
        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            map_path = m_config.map_path;
        }
        try
        {
            YAML::Node node = YAML::LoadFile(m_config.pose_save_path);

            // Reject the saved pose if the map file has been modified since it was saved.
            if (node["map_mtime"] && !map_path.empty() &&
                std::filesystem::exists(map_path))
            {
                auto saved_mtime  = node["map_mtime"].as<int64_t>();
                auto current_mtime = std::chrono::duration_cast<std::chrono::seconds>(
                    std::filesystem::last_write_time(map_path).time_since_epoch()).count();
                if (saved_mtime != current_mtime)
                {
                    RCLCPP_WARN(this->get_logger(),
                        "Saved pose is stale (map mtime changed). Discarding and running global search.");
                    return false;
                }
            }

            T = Eigen::Matrix4f::Identity();
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                {
                    std::string key = "m" + std::to_string(r) + std::to_string(c);
                    if (!node[key]) return false;
                    T(r, c) = node[key].as<float>();
                }
            return true;
        }
        catch (...) { return false; }
    }

    // ─── Timer: main localization loop ─────────────────────────────────────────
    void timerCB()
    {
        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            if (!m_state.message_received)
                return;
        }

        // In GLOBAL mode, handle state machine
        if (m_config.mode == LocalizerMode::GLOBAL)
        {
            globalModeUpdate();
            return;
        }

        // ICP mode (original behavior)
        icpModeUpdate();
    }

    // ─── ICP mode (original logic) ─────────────────────────────────────────────
    void icpModeUpdate()
    {
        rclcpp::Duration diff = this->get_clock()->now() - m_state.last_send_tf_time;
        bool update_tf = diff.seconds() > (1.0 / m_config.update_hz);

        if (!update_tf)
        {
            sendBroadCastTF();
            return;
        }

        m_state.last_send_tf_time = this->get_clock()->now();

        if (m_mapping_active.load())
        {
            sendBroadCastTF();
            return;
        }

        // A heavy service (relocalize map load, global search, map reload) may
        // be using the localizer objects; skip ICP this cycle and keep TF
        // alive from the last offset instead of blocking the timer.
        std::unique_lock<std::mutex> loc_lock(m_localizer_mutex, std::try_to_lock);
        if (!loc_lock.owns_lock())
        {
            sendBroadCastTF();
            return;
        }

        M4F initial_guess = M4F::Identity();
        M3D current_local_r;
        V3D current_local_t;
        builtin_interfaces::msg::Time current_time;
        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            if (m_state.service_received)
                initial_guess = m_state.initial_guess;
            else
            {
                initial_guess.block<3, 3>(0, 0) = (m_state.last_offset_r * m_state.last_r).cast<float>();
                initial_guess.block<3, 1>(0, 3) = (m_state.last_offset_r * m_state.last_t + m_state.last_offset_t).cast<float>();
            }
            current_local_r = m_state.last_r;
            current_local_t = m_state.last_t;
            current_time    = m_state.last_message_time;
            m_icp_localizer->setInput(m_state.last_cloud);
        }

        bool result = m_icp_localizer->align(initial_guess);
        loc_lock.unlock();
        if (result)
        {
            M3D map_body_r = initial_guess.block<3, 3>(0, 0).cast<double>();
            V3D map_body_t = initial_guess.block<3, 1>(0, 3).cast<double>();
            std::lock_guard<std::mutex> lock(m_state.mutex);
            m_state.last_offset_r = map_body_r * current_local_r.transpose();
            m_state.last_offset_t = -map_body_r * current_local_r.transpose() * current_local_t + map_body_t;
            if (!m_state.localize_success && m_state.service_received)
            {
                m_state.localize_success = true;
                m_state.service_received = false;
            }
        }
        sendBroadCastTF(current_time);
        publishMapCloud(current_time);
    }

    // ─── Global mode state machine ─────────────────────────────────────────────
    void globalModeUpdate()
    {
        // Reap a finished async search first, regardless of state, so the
        // running/done flags can never wedge if a service call moved the
        // state machine while the search was in flight.
        if (m_global_search_done.load())
            reapGlobalSearchResult();

        LocalizerState state;
        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            state = m_state.localizer_state;
        }

        switch (state)
        {
        case LocalizerState::IDLE:
            if (!m_global_search_running.load())
            {
                // Heavy services share the localizer objects; if one holds
                // them right now, retry next tick.
                std::unique_lock<std::mutex> loc_lock(m_localizer_mutex, std::try_to_lock);
                if (loc_lock.owns_lock() && m_global_localizer->isMapLoaded())
                {
                    CloudType::Ptr scan_copy(new CloudType);
                    size_t scan_pts = 0;
                    {
                        std::lock_guard<std::mutex> lock(m_state.mutex);
                        scan_pts = m_state.last_cloud->size();
                        if (static_cast<int>(scan_pts) >= m_config.min_cloud_points)
                            *scan_copy = *m_state.last_cloud;
                    }

                    // Don't attempt localization on a degenerate scan
                    if (static_cast<int>(scan_pts) < m_config.min_cloud_points)
                    {
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                            "Waiting for valid scan (%zu pts). Deferring localization.",
                            scan_pts);
                    }
                    else
                    {
                        // ── Try saved pose first (fast path for re-launches) ──
                        bool localized = false;
                        Eigen::Matrix4f prior;
                        if (loadPose(prior))
                        {
                            RCLCPP_INFO(this->get_logger(),
                                        "Found saved pose at %s. Trying prior ICP...",
                                        m_config.pose_save_path.c_str());
                            auto result = m_global_localizer->localizeFromPrior(scan_copy, prior);
                            if (result.has_value())
                            {
                                std::lock_guard<std::mutex> lock(m_state.mutex);
                                applyGlobalResultLocked(result.value(), m_state.last_r, m_state.last_t);
                                m_state.localizer_state  = LocalizerState::TRACKING;
                                m_state.localize_success = true;
                                localized = true;
                                RCLCPP_INFO(this->get_logger(),
                                            "Prior ICP succeeded — skipping global search.");
                            }
                            else
                                RCLCPP_WARN(this->get_logger(),
                                            "Prior ICP failed — falling back to global search.");
                        }

                        // ── Full global search ────────────────────────────────
                        if (!localized)
                        {
                            RCLCPP_INFO(this->get_logger(), "Starting global localization search...");
                            {
                                std::lock_guard<std::mutex> lock(m_state.mutex);
                                m_state.localizer_state = LocalizerState::SEARCHING;
                            }
                            // The async search thread takes the localizer mutex
                            // itself; release it here so it can start.
                            loc_lock.unlock();
                            startGlobalSearch();
                        }
                    }
                }
            }
            sendBroadCastTF();
            break;

        case LocalizerState::SEARCHING:
            // Keep broadcasting the last (or identity) TF while searching;
            // completion is handled by reapGlobalSearchResult() above.
            sendBroadCastTF();
            break;

        case LocalizerState::TRACKING:
        {
            rclcpp::Duration diff = this->get_clock()->now() - m_state.last_send_tf_time;
            bool update_tf = diff.seconds() > (1.0 / m_config.update_hz);

            if (!update_tf)
            {
                sendBroadCastTF();
                return;
            }
            m_state.last_send_tf_time = this->get_clock()->now();

            // ── Map-update mode: FAST_LIO takes over ──────────────────────────
            // The ICP offset is frozen at the trigger pose.  LIO odometry and
            // its own accumulation drive positioning from this point — exactly
            // like pure FAST_LIO mapping mode, anchored to the map frame.
            if (m_mapping_active.load())
            {
                builtin_interfaces::msg::Time t;
                { std::lock_guard<std::mutex> lk(m_state.mutex); t = m_state.last_message_time; }
                sendBroadCastTF(t);
                publishMapCloud(t);
                break;
            }

            // A heavy service (global reloc, map reload) is using the localizer
            // objects — keep TF alive from the last offset and skip this cycle.
            std::unique_lock<std::mutex> loc_lock(m_localizer_mutex, std::try_to_lock);
            if (!loc_lock.owns_lock())
            {
                sendBroadCastTF();
                break;
            }

            M4F initial_guess = M4F::Identity();
            builtin_interfaces::msg::Time current_time;
            M3D current_local_r;
            V3D current_local_t;
            size_t cloud_pts = 0;
            {
                std::lock_guard<std::mutex> lock(m_state.mutex);
                initial_guess.block<3, 3>(0, 0) = (m_state.last_offset_r * m_state.last_r).cast<float>();
                initial_guess.block<3, 1>(0, 3)  = (m_state.last_offset_r * m_state.last_t + m_state.last_offset_t).cast<float>();
                current_local_r = m_state.last_r;
                current_local_t = m_state.last_t;
                current_time    = m_state.last_message_time;
                cloud_pts       = m_state.last_cloud->size();
                // Only downsample / build KD-tree if the scan is worth using.
                // Avoids wasted voxel-filter CPU on degenerate scans.
                if (static_cast<int>(cloud_pts) >= m_config.min_cloud_points)
                    m_icp_localizer->setInput(m_state.last_cloud);
            }

            // Skip ICP on degenerate scans (lio_node "NO Effective Points" case)
            if (static_cast<int>(cloud_pts) < m_config.min_cloud_points)
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "Scan too sparse (%zu pts < %d). Holding last good pose.",
                    cloud_pts, m_config.min_cloud_points);
                loc_lock.unlock();
                sendBroadCastTF(current_time);
                publishMapCloud(current_time);
                break;
            }

            // Save predicted translation before ICP overwrites initial_guess.
            V3F predicted_t = initial_guess.block<3, 1>(0, 3);

            bool ok = m_icp_localizer->align(initial_guess);
            if (ok)
            {
                // Sanity check: ICP result must not jump far from the odometry
                // prediction.  A large jump (> icp_max_jump_m) means ICP
                // converged to a wrong local minimum — reject the result and
                // count it as a failure instead of corrupting the offset.
                float jump = (initial_guess.block<3, 1>(0, 3) - predicted_t).norm();
                if (jump > m_config.icp_max_jump_m)
                {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                        "ICP result jumped %.2f m from prediction (max %.2f m) — "
                        "likely wrong local minimum, rejecting.",
                        jump, m_config.icp_max_jump_m);
                    ok = false;
                }
            }
            if (ok)
            {
                M3D map_body_r = initial_guess.block<3, 3>(0, 0).cast<double>();
                V3D map_body_t = initial_guess.block<3, 1>(0, 3).cast<double>();
                int prev_failures = 0;
                {
                    std::lock_guard<std::mutex> lock(m_state.mutex);
                    m_state.last_offset_r = map_body_r * current_local_r.transpose();
                    m_state.last_offset_t = -map_body_r * current_local_r.transpose() * current_local_t + map_body_t;
                    prev_failures = m_state.consecutive_icp_failures;
                    m_state.consecutive_icp_failures = 0;
                }

                // Recovery widening was active — restore configured distances now
                // that tracking is healthy again.
                if (prev_failures > 0)
                    m_icp_localizer->setCorrespondenceDists(
                        m_icp_config.rough_max_corr_dist,
                        m_icp_config.refine_max_corr_dist);

                if (m_lio_dead_reckoning.load())
                {
                    RCLCPP_INFO(this->get_logger(),
                        "ICP re-acquired in mapped area — exiting LIO dead-reckoning mode.");
                    m_lio_dead_reckoning.store(false);
                }

                // Persist current map_body transform for next-session warm start
                rclcpp::Duration since_save = this->get_clock()->now() - m_state.last_pose_save_time;
                if (since_save.seconds() > 1.0 / m_config.pose_save_hz)
                {
                    savePose(initial_guess);
                    m_state.last_pose_save_time = this->get_clock()->now();
                }

                // Mapping accumulation is driven by worldCloudCB (sensor rate),
                // not here, so the ICP tracking path has zero extra overhead.
            }
            else
            {
                // ICP failed: count consecutive failures before declaring lost.
                int failures = 0;
                {
                    std::lock_guard<std::mutex> lock(m_state.mutex);
                    failures = ++m_state.consecutive_icp_failures;
                }
                const int max_failures = localizer_fsm::maxIcpFailures(
                    m_config.lost_patience_sec, m_config.update_hz);
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "ICP failure %d/%d", failures, max_failures);
                const localizer_fsm::IcpFailureDecision decision = localizer_fsm::onIcpFailure(
                    failures, max_failures, m_lio_dead_reckoning.load());

                // ── Intermediate recovery: widen correspondence distances at the
                //    halfway point so ICP can "reach" further before we give up
                //    and trigger a full global search.
                if (decision.widen_correspondence)
                {
                    double scale = m_icp_config.recovery_corr_scale;
                    m_icp_localizer->setCorrespondenceDists(
                        m_icp_config.rough_max_corr_dist  * scale,
                        m_icp_config.refine_max_corr_dist * scale);
                    RCLCPP_WARN(this->get_logger(),
                        "ICP recovery: widening correspondence distances by %.1fx (rough %.2f→%.2f, refine %.2f→%.2f)",
                        scale,
                        m_icp_config.rough_max_corr_dist,
                        m_icp_config.rough_max_corr_dist  * scale,
                        m_icp_config.refine_max_corr_dist,
                        m_icp_config.refine_max_corr_dist * scale);
                }

                if (decision.give_up)
                {
                    // Restore configured distances before triggering search
                    m_icp_localizer->setCorrespondenceDists(
                        m_icp_config.rough_max_corr_dist,
                        m_icp_config.refine_max_corr_dist);
                    {
                        std::lock_guard<std::mutex> lock(m_state.mutex);
                        m_state.consecutive_icp_failures = 0;
                    }

                    if (!decision.start_search)
                    {
                        // Already tried global search and it failed — robot is
                        // still in unmapped territory.  Keep dead-reckoning;
                        // don't spam the search thread.
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                            "Still in unmapped territory — LIO dead-reckoning active. "
                            "Drive back to mapped area or use 'Force Re-Localize'.");
                    }
                    else
                    {
                        RCLCPP_WARN(this->get_logger(),
                            "Lost localization after %d consecutive ICP failures. "
                            "Attempting global search.",
                            max_failures);
                        std::lock_guard<std::mutex> lock(m_state.mutex);
                        m_state.localizer_state = LocalizerState::IDLE;
                        m_state.localize_success = false;
                    }
                }
            }
            loc_lock.unlock();
            sendBroadCastTF(current_time);
            publishMapCloud(current_time);
            break;
        }

        case LocalizerState::LOST:
            sendBroadCastTF();
            // Transition back to IDLE to re-trigger search
            {
                std::lock_guard<std::mutex> lock(m_state.mutex);
                m_state.localizer_state = LocalizerState::IDLE;
            }
            break;
        }
    }

    // ─── Async search completion ────────────────────────────────────────────────
    // Runs on the timer thread once m_global_search_done is set. Applies the
    // result only if we are still SEARCHING and no forced service relocalization
    // superseded this search (generation check).
    void reapGlobalSearchResult()
    {
        m_global_search_done.store(false);
        m_global_search_running.store(false);

        if (m_global_search_result_gen != m_search_gen.load())
        {
            RCLCPP_WARN(this->get_logger(),
                "Discarding global search result superseded by a forced relocalization.");
            std::lock_guard<std::mutex> lock(m_state.mutex);
            if (m_state.localizer_state == LocalizerState::SEARCHING)
                m_state.localizer_state = LocalizerState::IDLE;
            return;
        }

        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            if (m_state.localizer_state != LocalizerState::SEARCHING)
                return; // a service call already re-localized us meanwhile
        }

        bool had_pose = false;
        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            had_pose = m_state.localize_success;
        }
        switch (localizer_fsm::searchOutcome(m_global_search_result.has_value(), had_pose))
        {
        case localizer_fsm::SearchOutcome::TRACK:
            {
                std::lock_guard<std::mutex> lock(m_state.mutex);
                // Use the snapshot odometry from when the search started so
                // that the offset is anchored to the correct moment in time.
                applyGlobalResultLocked(m_global_search_result.value(),
                                        m_global_search_snapshot_r,
                                        m_global_search_snapshot_t);
                m_state.localizer_state = LocalizerState::TRACKING;
                m_state.localize_success = true;
                m_state.consecutive_icp_failures = 0;
            }
            m_lio_dead_reckoning.store(false);
            RCLCPP_INFO(this->get_logger(), "Global localization succeeded! Switching to ICP tracking.");
            return;

        case localizer_fsm::SearchOutcome::DEAD_RECKON:
            // Robot HAD a good pose before — likely moved into unmapped
            // territory.  Hold LIO dead-reckoning rather than retrying (which
            // causes the IDLE→SEARCH→FAIL spin that glitches the position).
            // ICP re-acquires when the robot returns to mapped space.
            RCLCPP_WARN(this->get_logger(),
                "Global search failed — likely in unmapped territory. "
                "Holding LIO dead-reckoning. ICP will re-acquire in "
                "known areas. Use 'Force Re-Localize' to search manually.");
            {
                std::lock_guard<std::mutex> lock(m_state.mutex);
                m_state.localizer_state = LocalizerState::TRACKING;
                m_state.consecutive_icp_failures = 0;
            }
            m_lio_dead_reckoning.store(true);
            return;

        case localizer_fsm::SearchOutcome::RETRY:
        {
            // Robot has NEVER localized — dead-reckoning from identity would
            // be meaningless.  Keep retrying.
            RCLCPP_WARN(this->get_logger(),
                "Global search failed. Retrying initial localization...");
            std::lock_guard<std::mutex> lock(m_state.mutex);
            m_state.localizer_state = LocalizerState::IDLE;
            return;
        }
        }
    }

    void startGlobalSearch()
    {
        // Snapshot cloud, rotation AND translation at the moment the search
        // starts.  The search runs asynchronously and can take several seconds;
        // we must use these snapshot values (not the current odometry) when
        // computing the map↔local offset after the result arrives, otherwise
        // the predicted ICP initial_guess is stale by however far the robot
        // moved during the search.
        CloudType::Ptr scan_copy(new CloudType);
        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            *scan_copy = *m_state.last_cloud;
            m_global_search_snapshot_r = m_state.last_r;
            m_global_search_snapshot_t = m_state.last_t;
        }

        m_global_search_running.store(true);
        m_global_search_done.store(false);
        m_global_search_result = std::nullopt;
        uint64_t gen = m_search_gen.load();

        auto global_loc = m_global_localizer;
        Eigen::Matrix3f body_rot = m_global_search_snapshot_r.cast<float>();
        m_global_future = std::async(std::launch::async, [this, global_loc, scan_copy, body_rot, gen]() {
            std::optional<Eigen::Matrix4f> result;
            try
            {
                // Serialize against map reloads and service-triggered searches;
                // the tracking timer only try_locks this mutex, so TF stays alive.
                std::lock_guard<std::mutex> loc_lock(m_localizer_mutex);
                result = global_loc->localize(scan_copy, body_rot);
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(this->get_logger(), "Global search threw: %s", e.what());
            }
            // done (seq_cst) is set last so the timer thread sees result and
            // generation fully written once it observes done == true.
            m_global_search_result = result;
            m_global_search_result_gen = gen;
            m_global_search_done.store(true);
        });
    }

    // Apply a global localization result using the odometry pose (ref_r, ref_t)
    // captured when the scan given to the search was snapshotted.  Using the
    // snapshot rather than current odometry is essential: odometry keeps
    // updating while a search runs, and the offset must be anchored to the
    // moment the scan was taken.  Caller must hold m_state.mutex.
    void applyGlobalResultLocked(const Eigen::Matrix4f &T_map_body,
                                 const M3D &ref_r, const V3D &ref_t)
    {
        M3D map_body_r = T_map_body.block<3, 3>(0, 0).cast<double>();
        V3D map_body_t = T_map_body.block<3, 1>(0, 3).cast<double>();
        m_state.last_offset_r = map_body_r * ref_r.transpose();
        m_state.last_offset_t = -map_body_r * ref_r.transpose() * ref_t + map_body_t;
    }

    // ─── Sensor data callback ──────────────────────────────────────────────────
    // Updates the localization state (cloud + odometry) used by the ICP timer.
    // Mapping accumulation has moved to worldCloudCB so the full-resolution
    // LIO world cloud is used instead of the downsampled body cloud.
    void syncCB(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud_msg,
                const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg)
    {
        std::lock_guard<std::mutex> lock(m_state.mutex);
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

            if (m_config.mode == LocalizerMode::GLOBAL &&
                m_state.localizer_state == LocalizerState::IDLE)
            {
                RCLCPP_INFO(this->get_logger(), "First sensor data received. Global search will begin.");
            }
        }
    }

    // ─── TF broadcast ─────────────────────────────────────────────────────────
    // Snapshots the offset under the state mutex; callers must NOT hold it.
    void sendBroadCastTF(builtin_interfaces::msg::Time time)
    {
        geometry_msgs::msg::TransformStamped ts;
        Eigen::Quaterniond q;
        V3D t;
        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            ts.header.frame_id = m_config.map_frame;
            ts.child_frame_id  = m_config.local_frame;
            q = Eigen::Quaterniond(m_state.last_offset_r);
            t = m_state.last_offset_t;
        }
        ts.header.stamp       = time;
        ts.transform.translation.x = t.x();
        ts.transform.translation.y = t.y();
        ts.transform.translation.z = t.z();
        ts.transform.rotation.x = q.x();
        ts.transform.rotation.y = q.y();
        ts.transform.rotation.z = q.z();
        ts.transform.rotation.w = q.w();
        m_tf_broadcaster->sendTransform(ts);
    }

    // Broadcast with the stamp of the most recent sensor message.
    void sendBroadCastTF()
    {
        builtin_interfaces::msg::Time time;
        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            time = m_state.last_message_time;
        }
        sendBroadCastTF(time);
    }

    // ─── Service: manual relocalization with pose hint (original) ─────────────
    void relocCB(const std::shared_ptr<interface::srv::Relocalize::Request> request,
                 std::shared_ptr<interface::srv::Relocalize::Response> response)
    {
        if (!std::filesystem::exists(request->pcd_path))
        {
            response->success = false;
            response->message = "pcd file not found";
            return;
        }

        bool ok;
        {
            std::lock_guard<std::mutex> loc_lock(m_localizer_mutex);
            ok = m_icp_localizer->loadMap(request->pcd_path);
        }
        if (!ok)
        {
            response->success = false;
            response->message = "load map failed";
            return;
        }

        Eigen::AngleAxisd yaw_angle  = Eigen::AngleAxisd(request->yaw,   Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd roll_angle = Eigen::AngleAxisd(request->roll,  Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitch_angle= Eigen::AngleAxisd(request->pitch, Eigen::Vector3d::UnitY());
        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            m_state.initial_guess.setIdentity();
            m_state.initial_guess.block<3, 3>(0, 0) =
                (yaw_angle * roll_angle * pitch_angle).toRotationMatrix().cast<float>();
            m_state.initial_guess.block<3, 1>(0, 3) = V3F(request->x, request->y, request->z);
            m_state.service_received  = true;
            m_state.localize_success  = false;
        }
        response->success = true;
        response->message = "relocalize requested";
    }

    void relocCheckCB(const std::shared_ptr<interface::srv::IsValid::Request> request,
                      std::shared_ptr<interface::srv::IsValid::Response> response)
    {
        std::lock_guard<std::mutex> lock(m_state.mutex);
        response->valid = (request->code == 1) ? true : m_state.localize_success;
    }

    // ─── Service: global relocalization (no pose hint needed) ─────────────────
    // Runs in the heavy-service callback group: the search blocks this thread
    // only, while the timer keeps broadcasting TF from the last offset.
    void globalRelocCB(const std::shared_ptr<interface::srv::GlobalRelocalize::Request> request,
                       std::shared_ptr<interface::srv::GlobalRelocalize::Response> response)
    {
        std::string pcd_path = request->pcd_path;

        // If pcd_path is empty, use the currently loaded map
        if (!pcd_path.empty())
        {
            if (!std::filesystem::exists(pcd_path))
            {
                response->success = false;
                response->message = "pcd file not found: " + pcd_path;
                return;
            }
            std::lock_guard<std::mutex> loc_lock(m_localizer_mutex);
            bool ok1 = m_global_localizer->loadMap(pcd_path);
            bool ok2 = m_icp_localizer->loadMap(pcd_path);
            if (!ok1 || !ok2)
            {
                response->success = false;
                response->message = "Failed to load map";
                return;
            }
        }
        else
        {
            std::lock_guard<std::mutex> loc_lock(m_localizer_mutex);
            if (!m_global_localizer->isMapLoaded())
            {
                response->success = false;
                response->message = "No map loaded and no pcd_path provided";
                return;
            }
        }

        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            if (!m_state.message_received)
            {
                response->success = false;
                response->message = "No sensor data received yet";
                return;
            }
        }

        if (m_global_search_running.load())
        {
            if (!request->force)
            {
                response->success = false;
                response->message = "Global search already running. Set force=true to override.";
                return;
            }
            // Invalidate the in-flight async search: bump the generation so the
            // timer discards its result when it completes. The searches still
            // run one after the other (localizer mutex), never concurrently.
            m_search_gen.fetch_add(1);
        }

        // Snapshot scan AND odometry together: odometry keeps updating while
        // the search runs, and the resulting offset must be anchored to the
        // odometry pose that produced this scan.
        CloudType::Ptr scan_copy(new CloudType);
        M3D snapshot_r;
        V3D snapshot_t;
        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            *scan_copy = *m_state.last_cloud;
            snapshot_r = m_state.last_r;
            snapshot_t = m_state.last_t;
        }

        RCLCPP_INFO(this->get_logger(), "GlobalRelocalize service: running global search...");
        std::optional<Eigen::Matrix4f> result;
        double fitness = 0.0;
        {
            std::lock_guard<std::mutex> loc_lock(m_localizer_mutex);
            result = m_global_localizer->localize(scan_copy, snapshot_r.cast<float>());
            fitness = m_global_localizer->getLastFitness();
        }

        if (!result.has_value())
        {
            {
                // If we superseded an async search that also failed, don't
                // leave the state machine parked in SEARCHING.
                std::lock_guard<std::mutex> lock(m_state.mutex);
                if (m_state.localizer_state == LocalizerState::SEARCHING &&
                    !m_global_search_running.load())
                    m_state.localizer_state = LocalizerState::IDLE;
            }
            response->success = false;
            response->message = "Global localization failed";
            return;
        }

        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            applyGlobalResultLocked(result.value(), snapshot_r, snapshot_t);
            m_state.localizer_state = LocalizerState::TRACKING;
            m_state.localize_success = true;
            m_state.consecutive_icp_failures = 0;
        }
        m_lio_dead_reckoning.store(false);

        // Extract yaw from result for response
        Eigen::Matrix3f R = result.value().block<3, 3>(0, 0);
        double yaw_rad = std::atan2(R(1, 0), R(0, 0));

        response->success       = true;
        response->message       = "Global relocalization succeeded";
        response->x             = result.value()(0, 3);
        response->y             = result.value()(1, 3);
        response->z             = result.value()(2, 3);
        response->yaw_deg       = yaw_rad * 180.0 / M_PI;
        response->fitness_score = fitness;
    }

    // ─── Mapping accumulation ──────────────────────────────────────────────────
    // ─── Deferred scan_in_map publisher ───────────────────────────────────────
    // Publishes the full m_mapping_cloud (all points since "Start Map Update")
    // at 1 Hz in the map frame.  Because the cloud is already in map frame
    // (transformed in worldCloudCB) there is zero TF dependency — it cannot
    // drift or shift as ICP updates, which was the root cause of the "newly
    // accumulated map moves with the robot" symptom.
    void publishMappingCloud()
    {
        if (!m_mapping_active.load()) return;
        if (m_mapping_cloud_pub->get_subscription_count() < 1) return;
        sensor_msgs::msg::PointCloud2 msg;
        {
            std::lock_guard<std::mutex> lock(m_mapping_mutex);
            if (!m_mapping_cloud || m_mapping_cloud->empty()) return;
            pcl::toROSMsg(*m_mapping_cloud, msg);
        }
        msg.header.frame_id = m_config.map_frame;
        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            msg.header.stamp = m_state.last_message_time;
        }
        m_mapping_cloud_pub->publish(msg);
    }

    // ─── World cloud callback — mapping accumulation ───────────────────────────
    // Called every LiDAR sweep (~10 Hz).  When mapping is active the world_cloud
    // (in LIO lidar/world frame at scan_resolution) is transformed to map frame
    // using the ICP offset (map ← lidar), exactly matching the LIO odometry.
    void worldCloudCB(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
    {
        if (!m_mapping_active.load()) return;

        M4F T_map_lidar = M4F::Identity();

        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            // In GLOBAL mode require TRACKING; in ICP mode require at least one
            // successful localization (localize_success), since icpModeUpdate
            // never transitions localizer_state to TRACKING.
            bool ready = (m_config.mode == LocalizerMode::GLOBAL)
                ? (m_state.localizer_state == LocalizerState::TRACKING)
                : m_state.localize_success;
            if (!ready) return;

            // world_cloud is already in lidar (LIO world) frame, so the
            // transform to map frame is just the ICP offset: map ← lidar.
            T_map_lidar.block<3, 3>(0, 0) = m_state.last_offset_r.cast<float>();
            T_map_lidar.block<3, 1>(0, 3) = m_state.last_offset_t.cast<float>();
        }

        CloudType::Ptr cloud(new CloudType);
        pcl::fromROSMsg(*msg, *cloud);
        accumulateMappingCloud(cloud, T_map_lidar);
    }

    // ─── Mapping accumulation (called from worldCloudCB) ──────────────────────
    // The transform runs lock-free on a pre-snapshotted cloud + pose; only the
    // merge into m_mapping_cloud is serialized against stop_mapping.
    void accumulateMappingCloud(const CloudType::Ptr &cloud, const M4F &T_map_body)
    {
        CloudType::Ptr cloud_in_map(new CloudType);
        pcl::transformPointCloud(*cloud, *cloud_in_map, T_map_body);

        std::lock_guard<std::mutex> lock(m_mapping_mutex);
        // stop_mapping may have swapped the cloud out between the flag check
        // in worldCloudCB and here; don't accumulate into the fresh one.
        if (!m_mapping_active.load()) return;
        *m_mapping_cloud += *cloud_in_map;

        // Periodically voxel-filter to keep memory bounded.
        // Threshold at 5 M — at ~10 Hz world_cloud with ~10 k pts/scan this
        // fires at most every ~50 s, keeping filter cost well off the hot path.
        if (m_mapping_cloud->size() > 5000000)
        {
            CloudType::Ptr filtered(new CloudType);
            m_mapping_voxel.setInputCloud(m_mapping_cloud);
            m_mapping_voxel.filter(*filtered);
            m_mapping_cloud = filtered;
            RCLCPP_INFO(this->get_logger(),
                "[Mapping] Voxel-filtered mapping cloud → %zu pts", m_mapping_cloud->size());
        }
    }

    void startMappingCB(const std::shared_ptr<interface::srv::StartMapping::Request> request,
                        std::shared_ptr<interface::srv::StartMapping::Response> response)
    {
        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            if (m_config.mode == LocalizerMode::GLOBAL &&
                m_state.localizer_state != LocalizerState::TRACKING)
            {
                response->success = false;
                response->message = "Localization is not in TRACKING state. Cannot start mapping.";
                return;
            }
        }
        if (m_mapping_active.load())
        {
            response->success = false;
            response->message = "Mapping already active. Call stop_mapping first.";
            return;
        }

        float voxel = (request->voxel_size > 0.0f) ? request->voxel_size : 0.05f;
        {
            std::lock_guard<std::mutex> lock(m_mapping_mutex);
            m_mapping_output_path = request->output_path;
            m_mapping_voxel_leaf  = voxel;
            m_mapping_voxel.setLeafSize(voxel, voxel, voxel);
            m_mapping_cloud.reset(new CloudType);
        }
        m_mapping_active.store(true);

        RCLCPP_INFO(this->get_logger(),
            "[Mapping] Started. Output: %s, voxel: %.3f m",
            request->output_path.c_str(), voxel);
        response->success = true;
        response->message = "Mapping started";
    }

    // Runs in the heavy-service callback group: merge, filter, and save happen
    // on this thread with no locks held, so accumulation state is snapshotted
    // first and the timer keeps running throughout.
    void stopMappingCB(const std::shared_ptr<interface::srv::StopMapping::Request> request,
                       std::shared_ptr<interface::srv::StopMapping::Response> response)
    {
        if (!m_mapping_active.exchange(false))
        {
            response->success = false;
            response->message = "Mapping is not active.";
            return;
        }

        // Snapshot and reset the accumulation state; worldCloudCB stops
        // accumulating as soon as the flag above went false.
        CloudType::Ptr mapping_cloud;
        std::string start_output_path;
        float voxel_leaf;
        {
            std::lock_guard<std::mutex> lock(m_mapping_mutex);
            mapping_cloud = m_mapping_cloud;
            m_mapping_cloud.reset(new CloudType);
            start_output_path = m_mapping_output_path;
            voxel_leaf = m_mapping_voxel_leaf;
        }

        std::string map_path;
        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            map_path = m_config.map_path;
        }

        // ── Determine save path ────────────────────────────────────────────────
        // Priority: request override → start_mapping path → auto-name from loaded map
        std::string save_path = request->output_path.empty()
            ? start_output_path
            : request->output_path;

        if (save_path.empty())
        {
            // Auto-name: merged_<original_stem>.pcd in same directory
            if (map_path.empty())
            {
                response->success = false;
                response->message = "No output path and no loaded map to derive name from.";
                return;
            }
            std::filesystem::path orig(map_path);
            std::string stem = orig.stem().string();
            // Strip any leading "merged_" prefix so repeated stop→merge cycles
            // always save to merged_<original_stem>.pcd, not merged_merged_...
            if (stem.rfind("merged_", 0) == 0)
                stem = stem.substr(7);
            save_path = (orig.parent_path() / ("merged_" + stem)).string() + ".pcd";
        }

        // ── Merge with original map ────────────────────────────────────────────
        if (!map_path.empty() && std::filesystem::exists(map_path))
        {
            RCLCPP_INFO(this->get_logger(),
                "[Mapping] Merging new data with original map: %s", map_path.c_str());
            CloudType::Ptr original(new CloudType);
            if (pcl::io::loadPCDFile<PointType>(map_path, *original) >= 0)
                *mapping_cloud += *original;
            else
                RCLCPP_WARN(this->get_logger(),
                    "[Mapping] Could not load original map for merge — saving new data only.");
        }

        // ── Final voxel filter pass and save ──────────────────────────────────
        CloudType::Ptr final_cloud(new CloudType);
        pcl::VoxelGrid<PointType> voxel;
        voxel.setLeafSize(voxel_leaf, voxel_leaf, voxel_leaf);
        voxel.setInputCloud(mapping_cloud);
        voxel.filter(*final_cloud);

        if (pcl::io::savePCDFileBinaryCompressed(save_path, *final_cloud) < 0)
        {
            response->success = false;
            response->message = "Failed to save PCD to: " + save_path;
            return;
        }

        RCLCPP_INFO(this->get_logger(),
            "[Mapping] Saved merged map: %zu pts → %s", final_cloud->size(), save_path.c_str());

        // ── Hot-reload both localizers with the merged map ────────────────────
        // Done here on the service thread: while the localizer mutex is held,
        // the tracking timer skips ICP (try_lock) and keeps TF alive from the
        // last offset instead of stalling.
        RCLCPP_INFO(this->get_logger(),
            "[Mapping] Hot-reloading merged map into ICP + GlobalLocalizer: %s", save_path.c_str());
        bool icp_ok, global_ok;
        {
            std::lock_guard<std::mutex> loc_lock(m_localizer_mutex);
            icp_ok    = m_icp_localizer->loadMap(save_path);
            global_ok = m_global_localizer->loadMap(save_path);
        }
        if (icp_ok && global_ok)
        {
            {
                std::lock_guard<std::mutex> lock(m_state.mutex);
                m_config.map_path = save_path;
            }
            RCLCPP_INFO(this->get_logger(),
                "[Mapping] Map hot-reload complete. ICP continues tracking against new map.");
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(),
                "[Mapping] Map hot-reload FAILED (icp=%d global=%d). Still using old map.",
                icp_ok, global_ok);
        }

        response->success    = true;
        response->message    = "Merged map saved";
        response->point_count= static_cast<int32_t>(final_cloud->size());
        response->saved_path = save_path;
    }

    // ─── Status publisher (1 Hz) ───────────────────────────────────────────────
    // Machine-parseable one-liner for GUIs/monitoring:
    //   mode=GLOBAL state=TRACKING dead_reckoning=0 mapping=0 searching=0
    void publishStatus()
    {
        if (m_status_pub->get_subscription_count() < 1)
            return;
        LocalizerState state;
        {
            std::lock_guard<std::mutex> lock(m_state.mutex);
            state = m_state.localizer_state;
        }
        const char *state_str = "IDLE";
        switch (state)
        {
        case LocalizerState::IDLE:      state_str = "IDLE";      break;
        case LocalizerState::SEARCHING: state_str = "SEARCHING"; break;
        case LocalizerState::TRACKING:  state_str = "TRACKING";  break;
        case LocalizerState::LOST:      state_str = "LOST";      break;
        }
        std_msgs::msg::String msg;
        msg.data = std::string("mode=") +
                   (m_config.mode == LocalizerMode::GLOBAL ? "GLOBAL" : "ICP") +
                   " state=" + state_str +
                   " dead_reckoning=" + (m_lio_dead_reckoning.load() ? "1" : "0") +
                   " mapping=" + (m_mapping_active.load() ? "1" : "0") +
                   " searching=" + (m_global_search_running.load() ? "1" : "0");
        m_status_pub->publish(msg);
    }

    // ─── Map cloud publisher ───────────────────────────────────────────────────
    // refineMap() hands back the ICP localizer's internal cloud, which loadMap()
    // mutates in place — serialize against reloads while it is read. Callers
    // must not hold m_localizer_mutex.
    void publishMapCloud(builtin_interfaces::msg::Time time)
    {
        if (m_map_cloud_pub->get_subscription_count() < 1)
            return;
        sensor_msgs::msg::PointCloud2 msg;
        {
            std::unique_lock<std::mutex> loc_lock(m_localizer_mutex, std::try_to_lock);
            if (!loc_lock.owns_lock())
                return; // map being reloaded/searched; publish next cycle
            CloudType::Ptr map_cloud = m_icp_localizer->refineMap();
            if (!map_cloud || map_cloud->empty())
                return;
            pcl::toROSMsg(*map_cloud, msg);
        }
        msg.header.frame_id = m_config.map_frame;
        msg.header.stamp    = time;
        m_map_cloud_pub->publish(msg);
    }

private:
    NodeConfig  m_config;
    NodeState   m_state;
    ICPConfig   m_icp_config;
    GlobalLocConfig m_global_config;

    std::shared_ptr<ICPLocalizer>    m_icp_localizer;
    std::shared_ptr<GlobalLocalizer> m_global_localizer;

    // Serializes ALL use of m_icp_localizer / m_global_localizer — their
    // internals (PCL ICP/NDT objects, map clouds mutated in place by loadMap)
    // are not thread-safe. Held by: the tracking timer (try_lock only, so TF
    // never stalls), the async search thread, and the heavy services.
    // Lock order: m_localizer_mutex before m_state.mutex, never the reverse.
    std::mutex m_localizer_mutex;

    // True when global search failed (unmapped area) and we are holding the
    // map pose steady using raw LIO odometry integration.  Cleared when ICP
    // re-acquires in a mapped region.
    std::atomic<bool>                 m_lio_dead_reckoning{false};

    // Global search async state
    std::atomic<bool>                 m_global_search_running{false};
    std::atomic<bool>                 m_global_search_done{false};
    std::optional<Eigen::Matrix4f>    m_global_search_result;
    std::future<void>                 m_global_future;
    // Generation counter: a forced service relocalization bumps m_search_gen
    // so the timer discards the result of a superseded in-flight search.
    // m_global_search_result_gen is written by the async thread before the
    // done flag, so it is safe to read once done is observed true.
    std::atomic<uint64_t>             m_search_gen{0};
    uint64_t                          m_global_search_result_gen = 0;
    // Odometry snapshot taken at the moment the async search was launched.
    // Must be used (not current odometry) when applying the search result,
    // because the robot may have moved significantly during the search.
    M3D                               m_global_search_snapshot_r = M3D::Identity();
    V3D                               m_global_search_snapshot_t = V3D::Zero();

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> m_cloud_sub;
    message_filters::Subscriber<nav_msgs::msg::Odometry>       m_odom_sub;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> m_sync;

    rclcpp::TimerBase::SharedPtr m_timer;
    std::shared_ptr<tf2_ros::TransformBroadcaster>             m_tf_broadcaster;

    // World cloud subscription for map-update accumulation (LIO full resolution)
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_world_cloud_sub;

    rclcpp::Service<interface::srv::Relocalize>::SharedPtr       m_reloc_srv;
    rclcpp::Service<interface::srv::IsValid>::SharedPtr          m_reloc_check_srv;
    rclcpp::Service<interface::srv::GlobalRelocalize>::SharedPtr m_global_reloc_srv;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr  m_map_cloud_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr  m_mapping_cloud_pub;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr          m_status_pub;
    rclcpp::TimerBase::SharedPtr                                 m_status_timer;

    // Mapping accumulation. The flag is atomic (checked on the sensor path);
    // cloud, voxel grid, and paths are guarded by m_mapping_mutex because
    // stop_mapping snapshots them from the heavy-service thread.
    std::atomic<bool>   m_mapping_active{false};
    std::mutex          m_mapping_mutex;
    std::string         m_mapping_output_path;
    CloudType::Ptr      m_mapping_cloud{new CloudType};
    pcl::VoxelGrid<PointType> m_mapping_voxel;
    float               m_mapping_voxel_leaf = 0.05f;

    // 1 Hz timer that re-publishes the full m_mapping_cloud in map frame
    rclcpp::TimerBase::SharedPtr  m_mapping_cloud_timer;

    // Heavy services (relocalize, global_relocalize, stop_mapping) run here so
    // the MultiThreadedExecutor can keep the timer/TF alive while they work.
    rclcpp::CallbackGroup::SharedPtr m_heavy_cb_group;

    rclcpp::Service<interface::srv::StartMapping>::SharedPtr m_start_mapping_srv;
    rclcpp::Service<interface::srv::StopMapping>::SharedPtr  m_stop_mapping_srv;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    int ret = 0;
    try
    {
        auto node = std::make_shared<LocalizerNode>();
        // Two threads: sensor callbacks + localization timer stay in the default
        // (mutually exclusive) group; heavy services run on the other thread so a
        // global search or map merge/save never stalls the map→local TF.
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);
        executor.spin();
    }
    catch (const std::exception &e)
    {
        RCLCPP_FATAL(rclcpp::get_logger("localizer_node"), "%s", e.what());
        ret = 1;
    }
    rclcpp::shutdown();
    return ret;
}
