#include "lidar_refinement.hpp"

#include <aduulm_object_types_lib/object_access.hpp>

#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/utils.h>  // has to be included after tf2_geometry_msgs!

#include "logger_setup.hpp"

DEFINE_LOGGER_VARIABLES

namespace capsule_pose_estimation
{

static constexpr auto DEFAULT_MARKER_TOPIC = "marker_detections";
static constexpr auto DEFAULT_LIDAR_TOPIC = "points";
static constexpr auto DEFAULT_OUTPUT_TOPIC = "refined_marker_detections";

static geometry_msgs::msg::Pose poseFromDetection(const lib::Detection& detection);

LidarMarkerRefinement::LidarMarkerRefinement(const std::string& name)
  : rclcpp_lifecycle::LifecycleNode{ name,
                                     rclcpp::NodeOptions()
                                         .allow_undeclared_parameters(true)
                                         .automatically_declare_parameters_from_overrides(true) }
  , param_handler_(std::make_unique<ParameterHandler>(this->get_node_parameters_interface(), this))
  , tf_buffer_(get_clock())
  , tf_listener_(tf_buffer_)
  , last_pcl_(std::make_shared<pcl::PointCloud<AduulmPoint>>())
  , params_(std::make_shared<lib::lidar::RefinementParams>())
{
  initLogger();
}

LidarMarkerRefinement::CallbackReturn LidarMarkerRefinement::on_configure(const rclcpp_lifecycle::State& state)
{
  LOG_INF("configuring...");
  handleParameter();

  advertiseTopics();

  refinement_toolbox_ = std::make_unique<lib::lidar::RefinementToolbox>(params_);

  LOG_INF("done configuring");
  return CallbackReturn::SUCCESS;
}

LidarMarkerRefinement::CallbackReturn LidarMarkerRefinement::on_activate(const rclcpp_lifecycle::State& state)
{
  LOG_INF("activating...");
  refined_marker_pub_->on_activate();
  debug_cloud_publisher_->on_activate();

  LOG_INF("done activating");
  return CallbackReturn::SUCCESS;
}

LidarMarkerRefinement::CallbackReturn LidarMarkerRefinement::on_deactivate(const rclcpp_lifecycle::State& state)
{
  refined_marker_pub_->on_deactivate();
  debug_cloud_publisher_->on_deactivate();

  return CallbackReturn::SUCCESS;
}

LidarMarkerRefinement::CallbackReturn LidarMarkerRefinement::on_cleanup(const rclcpp_lifecycle::State& state)
{
  return CallbackReturn::SUCCESS;
}

LidarMarkerRefinement::CallbackReturn LidarMarkerRefinement::on_shutdown(const rclcpp_lifecycle::State& state)
{
  return CallbackReturn::SUCCESS;
}

void LidarMarkerRefinement::initLogger()
{
  LOGGER_ADD_SUBLOGGER_LIBRARY(capsule_pose_estimation::lib);
  _initLogger();
  _setStreamName("LidarMarkerRefinement");
  _setLogLevel(aduulm_logger::LoggerLevel::Warn);
}

bool LidarMarkerRefinement::handleParameter()
{
  LOG_DEB("Handling parameters");

  param_handler_->handleDynamicParameter<double>("filter_distance.search_dist_next_marker",
                                                 params_->filter_distance.search_dist_next_marker,
                                                 params_->filter_distance.search_dist_next_marker);
  param_handler_->handleDynamicParameter<double>("filter_distance.search_dist_depth_marker",
                                                 params_->filter_distance.search_dist_depth_marker,
                                                 params_->filter_distance.search_dist_depth_marker);
  param_handler_->handleDynamicParameter<double>("filter_distance.upper_z",
                                                 params_->filter_distance.upper_filtering_distance_z,
                                                 params_->filter_distance.upper_filtering_distance_z);
  param_handler_->handleDynamicParameter<double>("filter_distance.lower_z",
                                                 params_->filter_distance.lower_filtering_distance_z,
                                                 params_->filter_distance.lower_filtering_distance_z);

  param_handler_->handleDynamicParameter<double>("ransac_params.ransac_threshold",
                                                 params_->ransac_params.ransac_threshold,
                                                 params_->ransac_params.ransac_threshold);
  param_handler_->handleDynamicParameter<bool>("ransac_params.optimize_coefficients",
                                               params_->ransac_params.optimize_coefficients,
                                               params_->ransac_params.optimize_coefficients);
  param_handler_->handleDynamicParameter<int>(
      "ransac_params.method_type",
      [&](int value) {
        if (value < 0 or value > 6)
        {
          LOG_ERR("Only values between 0..6 are allowed. They are:\n"
                  "SAC_RANSAC  = 0\n"
                  "SAC_LMEDS   = 1\n"
                  "SAC_MSAC    = 2\n"
                  "SAC_RRANSAC = 3\n"
                  "SAC_RMSAC   = 4\n"
                  "SAC_MLESAC  = 5\n"
                  "SAC_PROSAC  = 6");
        }
        else
        {
          params_->ransac_params.method_type = value;
        }
      },
      params_->ransac_params.method_type);

  param_handler_->handleDynamicParameter<double>("min_distance", params_->min_distance, params_->min_distance);
  param_handler_->handleDynamicParameter<bool>(
      "prepare_debug_pcl", params_->prepare_debug_pcl, params_->prepare_debug_pcl);
  param_handler_->handleDynamicParameter<double>(
      "uncertainty_reduction", params_->uncertainty_reduction, params_->uncertainty_reduction);
  param_handler_->handleDynamicParameter<bool>("refine_boards", params_->refine_boards, params_->refine_boards);

  param_handler_->handleLoggerLevelParameter(
      "log_level",
      [&](aduulm_logger::LoggerLevel log_level) { _setLogLevel(log_level); },
      aduulm_logger::LoggerLevel::Info);

  LOG_DEB("Handled all parameters");

  return true;
}

bool LidarMarkerRefinement::advertiseTopics()
{
  lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      DEFAULT_LIDAR_TOPIC, 1, [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { this->lidarCallback(msg); });

  marker_sub_ = this->create_subscription<aduulm_object_interfaces::msg::DetectionList>(
      DEFAULT_MARKER_TOPIC, 1, [this](aduulm_object_interfaces::msg::DetectionList::ConstSharedPtr msg) {
        this->markerCallback(msg);
      });

  refined_marker_pub_ = this->create_publisher<aduulm_object_interfaces::msg::DetectionList>(DEFAULT_OUTPUT_TOPIC, 1);

  debug_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug_pointcloud", 1);

  return true;
}

void LidarMarkerRefinement::lidarCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  // Note: To keep things simple we simply cache the last point cloud message and match it with the next received
  //       marker. Of course a proper matching (e.g. via the minimal latency buffer) could reduce the time difference
  //       between lidar and camera, however, it is assumed that we drive rather slow anyway

  LOG_DEB_THROTTLE(1.0, "Point cloud received");

  // we assume a single threaded executor, hence, overriding the point cloud should be safe
  pcl::fromROSMsg(*msg, *last_pcl_);
  lidar_frame_id_ = msg->header.frame_id;
}

void LidarMarkerRefinement::markerCallback(aduulm_object_interfaces::msg::DetectionList::ConstSharedPtr msg)
{
  LOG_DEB_THROTTLE(1.0, "Received " << msg->detections.size() << " camera markers");

  if (lidar_frame_id_.empty())
  {
    LOG_ERR("lidar frame_id is empty!");
    return;
  }

  // determine transformation from marker to lidar frame
  std::optional<geometry_msgs::msg::TransformStamped> marker_to_lidar_trafo;
  std::optional<geometry_msgs::msg::TransformStamped> camera_to_lidar_trafo;

  try
  {
    marker_to_lidar_trafo = tf_buffer_.lookupTransform(lidar_frame_id_, msg->header.frame_id, msg->header.stamp);
  }
  catch (const tf2::ExtrapolationException& ex)
  {
    marker_to_lidar_trafo = tf_buffer_.lookupTransform(lidar_frame_id_, msg->header.frame_id, tf2::TimePointZero);
  }
  catch (const tf2::TransformException& ex)
  {
    LOG_ERR("Unable to retrieve transformation from '" << msg->header.frame_id << "' to '" << lidar_frame_id_ << "'");
  }

  // extract camera sensor frame from meta info
  std::string camera_sensor_frame;
  if (!msg->meta_info.empty())
  {
    for (const auto& elem : msg->meta_info)
    {
      if (elem.rfind("frame_id:") == 0)
      {
        camera_sensor_frame = elem.substr(9);
      }
    }
  }

  if (not camera_sensor_frame.empty())
  {
    try
    {
      camera_to_lidar_trafo = tf_buffer_.lookupTransform(lidar_frame_id_, camera_sensor_frame, msg->header.stamp);
    }
    catch (const tf2::ExtrapolationException& ex)
    {
      camera_to_lidar_trafo = tf_buffer_.lookupTransform(lidar_frame_id_, camera_sensor_frame, tf2::TimePointZero);
    }
    catch (const tf2::TransformException& ex)
    {
      LOG_ERR("Unable to retrieve transformation from '" << camera_sensor_frame << "' to '" << lidar_frame_id_ << "'");
    }
  }

  if (!marker_to_lidar_trafo.has_value())
  {
    // couldn't refine markers --> forward input
    refined_marker_pub_->publish(*msg);
    return;
  }

  // transform detections and distinguish between board (won't be refined) and normal markers
  std::vector<lib::Detection> markers;

  for (const auto& detection : msg->detections)
  {
    geometry_msgs::msg::PoseStamped pose_msg;
    geometry_msgs::msg::PoseStamped out_pose_msg;
    pose_msg.header.stamp = msg->header.stamp;
    pose_msg.pose.position.x = detection.state.mean[0];
    pose_msg.pose.position.y = detection.state.mean[1];
    pose_msg.pose.position.z = detection.state.mean[2];
    tf2::Quaternion quat;
    const double yaw = detection.state.mean[3];
    quat.setRPY(0, 0, yaw);
    pose_msg.pose.orientation = tf2::toMsg(quat);

    // get covariance from detecton
    namespace ac = aduulm_object_types_lib;
    namespace Attr = ac::Attributes;
    Eigen::Matrix<double, 6, 6> cov = Eigen::MatrixXd::Zero(6, 6);  // empty default matrix
    // insert covariance into top left corner, ignoring pitch and roll attributes
    const std::vector attributes = { ac::Attributes::X, ac::Attributes::Y, ac::Attributes::Z, ac::Attributes::YAW };
    cov.topLeftCorner<4, 4>() = ac::getCovarianceMatrix(detection.state, attributes);
    // insert flattened array
    std::array<double, 36> flat_cov;
    Eigen::Matrix<double, 6, 6>::Map(flat_cov.data()) = cov;

    // apply transformation
    tf2::doTransform(pose_msg, out_pose_msg, marker_to_lidar_trafo.value());

    // create tf2::transform for cov
    tf2::Transform tf2_transform(tf2::Quaternion(marker_to_lidar_trafo.value().transform.rotation.x,
                                                 marker_to_lidar_trafo.value().transform.rotation.y,
                                                 marker_to_lidar_trafo.value().transform.rotation.z,
                                                 marker_to_lidar_trafo.value().transform.rotation.w),
                                 tf2::Vector3(marker_to_lidar_trafo.value().transform.translation.x,
                                              marker_to_lidar_trafo.value().transform.translation.y,
                                              marker_to_lidar_trafo.value().transform.translation.z));
    flat_cov = tf2::transformCovariance(flat_cov, tf2_transform);
    cov = Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(flat_cov.data());

    bool refined_by_board = false;
    double marker_angle_to_camera_ray = 0.0;
    for (const auto& info : detection.meta_info)
    {
      if (info == "refined")
      {
        refined_by_board = true;
        continue;
      }
      if (info.find("marker_angle_to_camera_ray") != std::string::npos)
      {
        marker_angle_to_camera_ray = stod(info.substr(info.find(":") + 1));
        continue;
      }
    }

    LOG_DEB("Added marker to lib");
    // convert into lib data format
    markers.emplace_back(out_pose_msg.pose.position.x,
                         out_pose_msg.pose.position.y,
                         out_pose_msg.pose.position.z,
                         out_pose_msg.pose.orientation.x,
                         out_pose_msg.pose.orientation.y,
                         out_pose_msg.pose.orientation.z,
                         out_pose_msg.pose.orientation.w,
                         cov.topLeftCorner<2, 2>(),
                         detection.id,
                         marker_angle_to_camera_ray,
                         refined_by_board);
  }

  std::optional<lib::lidar::Pos3D> camera_pos = lib::lidar::Pos3D();  // provided in lidar coordinates
  if (camera_to_lidar_trafo.has_value())
  {
    camera_pos->x = static_cast<float>(camera_to_lidar_trafo->transform.translation.x);
    camera_pos->y = static_cast<float>(camera_to_lidar_trafo->transform.translation.y);
    camera_pos->z = static_cast<float>(camera_to_lidar_trafo->transform.translation.z);
  }

  const std::vector<lib::Detection> refined_markers =
      refinement_toolbox_->refineMarkers(markers, last_pcl_, camera_sensor_frame, camera_pos);

  publishRefinedMarkers(msg, refined_markers, lidar_frame_id_);

  if (params_->prepare_debug_pcl)
  {
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(refinement_toolbox_->getDebugPCL(), output);
    output.header.stamp = msg->header.stamp;
    output.header.frame_id = lidar_frame_id_;
    debug_cloud_publisher_->publish(output);
  }
}

void LidarMarkerRefinement::publishRefinedMarkers(aduulm_object_interfaces::msg::DetectionList::ConstSharedPtr msg,
                                                  const std::vector<lib::Detection>& refined_markers,
                                                  const std::string& lidar_frame_id)
{
  // Note: in order to avoid transforming the uncertainty we transform all markers back into the input coordinate frame
  aduulm_object_interfaces::msg::DetectionList output_msg;
  output_msg.header = msg->header;

  std::optional<geometry_msgs::msg::TransformStamped> lidar2marker;
  try
  {
    lidar2marker = tf_buffer_.lookupTransform(msg->header.frame_id, lidar_frame_id, msg->header.stamp);
  }
  catch (tf2::ExtrapolationException& ex)
  {
    lidar2marker = tf_buffer_.lookupTransform(msg->header.frame_id, lidar_frame_id, tf2::TimePointZero);
  }
  catch (tf2::TransformException& ex)
  {
    LOG_ERR(ex.what());
  }

  if (lidar2marker.has_value())
  {
    for (const auto& refined_detection : refined_markers)
    {
      // search for matching detection in camera only detections and refine it
      std::optional<aduulm_object_interfaces::msg::Detection> matching_detection;
      for (const auto& detection_msg : msg->detections)
      {
        if (detection_msg.id == refined_detection.id)
        {
          matching_detection = detection_msg;
        }
      }

      if (not matching_detection.has_value())
      {
        continue;
      }

      auto pose = poseFromDetection(refined_detection);
      tf2::doTransform(pose, pose, lidar2marker.value());
      using MsgStateType = aduulm_object_types_lib::CP3D_Y;
      auto object_state = aduulm_object_types_lib::wrap<MsgStateType>(matching_detection->state);

      object_state.attr(MsgStateType::X) = pose.position.x;
      object_state.attr(MsgStateType::Y) = pose.position.y;
      object_state.attr(MsgStateType::Z) = pose.position.z;

      tf2::Quaternion pose_quat;
      tf2::fromMsg(pose.orientation, pose_quat);
      object_state.attr(MsgStateType::YAW) = tf2::getYaw(pose_quat);

      if (refined_detection.refined_by_lidar)
      {
        // assumes output happens in some frame with z-axis facing upwards
        // --> goal: circular uncertainty ellipsis with reduced size (due to improved distance measurement)
        auto min_variance = std::min(object_state.var(MsgStateType::X), object_state.var(MsgStateType::Y));

        min_variance *= params_->uncertainty_reduction;

        object_state.var(MsgStateType::X) = min_variance;
        object_state.var(MsgStateType::Y) = min_variance;
        object_state.var(MsgStateType::YAW) = object_state.var(MsgStateType::YAW) * params_->uncertainty_reduction;

        object_state.cov(MsgStateType::X, MsgStateType::Y) = 0.0;
        object_state.cov(MsgStateType::Y, MsgStateType::X) = 0.0;
        matching_detection->meta_info.emplace_back("lidar_refined");
      }
      output_msg.detections.push_back(matching_detection.value());
    }
  }

  refined_marker_pub_->publish(output_msg);
}

geometry_msgs::msg::Pose poseFromDetection(const lib::Detection& detection)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = detection.x;
  pose.position.y = detection.y;
  pose.position.z = detection.z;

  pose.orientation.x = detection.rotation_x;
  pose.orientation.y = detection.rotation_y;
  pose.orientation.z = detection.rotation_z;
  pose.orientation.w = detection.rotation_w;

  return pose;
}

DEFINE_LOGGER_CLASS_INTERFACE_IMPLEMENTATION(LidarMarkerRefinement)
}  // namespace capsule_pose_estimation

int main(int argc, char** argv)
{
  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  rclcpp::init(argc, argv);

  auto node = std::make_shared<capsule_pose_estimation::LidarMarkerRefinement>("lidar_marker_refinement");

  CallbackReturn configure_return;
  node->configure(configure_return);
  if (CallbackReturn::SUCCESS == configure_return)
  {
    node->activate();
  }

  rclcpp::executors::SingleThreadedExecutor exe;
  exe.add_node(node->get_node_base_interface());

  exe.spin();

  rclcpp::shutdown();

  return 0;
}