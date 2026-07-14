/**
 * @file capsule_pose_estimation.cpp
 *
 * @brief Node for the pose estimation of U-Shift II capsules
 *
 * @author Oliver Schumann
 * Contact: oliver.schumann@uni-ulm.de
 *
 */
#include "camera_detection.hpp"

#include <map>
#include <utility>
#include <cv_bridge/cv_bridge.hpp>

#include <aduulm_object_types_lib/object_access.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/utils.h>  // has to be included after tf2_geometry_msgs!

#include "logger_setup.hpp"

DEFINE_LOGGER_VARIABLES

namespace capsule_pose_estimation
{
static constexpr auto DEFAULT_CPU_IMAGE_TOPIC = "image_rect";
static constexpr auto DEFAULT_CPU_IMAGE_INFO_TOPIC = "camera_info";
static constexpr auto DEFAULT_GPU_IMAGE_TOPIC = "gpu_image_rect";
static constexpr auto DEFAULT_DEBUG_IMAGE_TOPIC = "debug_image";
static constexpr auto DEFAULT_MARKER_DETECTION_TOPIC = "marker_detections";

// we expect the detector to estimate at least the right quadrant
static constexpr auto MAX_ORIENTATION_STDDEV_RAD = M_PI_4;

static const auto TRANSFORM_TIMEOUT = rclcpp::Duration(1, 0);

static std::map<std::string, cv::aruco::PredefinedDictionaryType> dict_mappings = {
  { "4x4_50", cv::aruco::PredefinedDictionaryType::DICT_4X4_50 },
  { "4x4_100", cv::aruco::PredefinedDictionaryType::DICT_4X4_100 },
  { "4x4_250", cv::aruco::PredefinedDictionaryType::DICT_4X4_250 },
  { "4x4_1000", cv::aruco::PredefinedDictionaryType::DICT_4X4_1000 },
  { "5x5_50", cv::aruco::PredefinedDictionaryType::DICT_5X5_50 },
  { "5x5_100", cv::aruco::PredefinedDictionaryType::DICT_5X5_100 },
  { "5x5_250", cv::aruco::PredefinedDictionaryType::DICT_5X5_250 },
  { "5x5_1000", cv::aruco::PredefinedDictionaryType::DICT_5X5_1000 },
  { "6x6_50", cv::aruco::PredefinedDictionaryType::DICT_6X6_50 },
  { "6x6_100", cv::aruco::PredefinedDictionaryType::DICT_6X6_100 },
  { "6x6_250", cv::aruco::PredefinedDictionaryType::DICT_6X6_250 },
  { "6x6_1000", cv::aruco::PredefinedDictionaryType::DICT_6X6_1000 },
  { "7x7_50", cv::aruco::PredefinedDictionaryType::DICT_7X7_50 },
  { "7x7_100", cv::aruco::PredefinedDictionaryType::DICT_7X7_100 },
  { "7x7_250", cv::aruco::PredefinedDictionaryType::DICT_7X7_250 },
  { "7x7_1000", cv::aruco::PredefinedDictionaryType::DICT_7X7_1000 },
};

// declaration of utility functions
[[nodiscard]] cv::Mat extractDistortionCoefficients(const sensor_msgs::msg::CameraInfo& msg);
[[nodiscard]] cv::Mat extractCameraMatrix(const sensor_msgs::msg::CameraInfo& msg);
[[nodiscard]] geometry_msgs::msg::Pose poseFromDetection(const lib::Detection& detection);

CapsulePoseEstimation::CapsulePoseEstimation(const std::string& name)
  : rclcpp_lifecycle::LifecycleNode{ name,
                                     rclcpp::NodeOptions()
                                         .allow_undeclared_parameters(true)
                                         .automatically_declare_parameters_from_overrides(true) }
  , debug_params_(std::make_shared<lib::DebuggingParameters>())
  , param_handler_(std::make_unique<ParameterHandler>(this->get_node_parameters_interface(), this))
  , tf_buffer_(get_clock())
  , tf_listener_(tf_buffer_)
  , tf_broadcaster_(std::make_shared<tf2_ros::TransformBroadcaster>(this))
{
  initLogger();
}

CapsulePoseEstimation::CallbackReturn CapsulePoseEstimation::on_configure(const rclcpp_lifecycle::State& state)
{
  if (!handleParameter())
  {
    return CallbackReturn::FAILURE;
  }

  advertiseTopics();

  toolbox_ = std::make_unique<lib::ArucoToolbox>(params_, debug_params_);

  LOG_INF("done configuring");

  return CallbackReturn::SUCCESS;
}

CapsulePoseEstimation::CallbackReturn CapsulePoseEstimation::on_activate(const rclcpp_lifecycle::State& state)
{
  LOG_INF("activating...");
  debug_image_pub_->on_activate();
  marker_detection_pub_->on_activate();
  LOG_INF("done activating");
  return CallbackReturn::SUCCESS;
}

CapsulePoseEstimation::CallbackReturn CapsulePoseEstimation::on_deactivate(const rclcpp_lifecycle::State& state)
{
  debug_image_pub_->on_deactivate();
  marker_detection_pub_->on_deactivate();
  return CallbackReturn::SUCCESS;
}

CapsulePoseEstimation::CallbackReturn CapsulePoseEstimation::on_cleanup(const rclcpp_lifecycle::State& state)
{
  return CallbackReturn::SUCCESS;
}

CapsulePoseEstimation::CallbackReturn CapsulePoseEstimation::on_shutdown(const rclcpp_lifecycle::State& state)
{
  return CallbackReturn::SUCCESS;
}

void CapsulePoseEstimation::initLogger()
{
  _initLogger();
  _setStreamName("CapsulePoseEstimation");
  _setLogLevel(aduulm_logger::LoggerLevel::Warn);
  LOGGER_ADD_SUBLOGGER_LIBRARY(capsule_pose_estimation::lib);
}

bool CapsulePoseEstimation::handleParameter()
{
  param_handler_->handleStaticParameter<std::string>("output_frame_id", output_frame_id_);
  param_handler_->handleStaticParameter<float>("marker_size", params_.marker_size);
  param_handler_->handleStaticParameter<bool>("use_gpu", use_gpu_);

  if (use_gpu_)
  {
    param_handler_->handleStaticParameter<std::string>("gpu_shared_mem_service_ns", gpu_shared_mem_services_ns_);
  }

  // Improvement note: Rejecting invalid parameters directly would be nice, but is currently (02/2024) not
  //                   implemented within the parameter handler.
  std::string new_aruco_dict;
  param_handler_->handleStaticParameter<std::string>("aruco_dict", new_aruco_dict);

  if (dict_mappings.find(new_aruco_dict) != dict_mappings.end())
  {
    aruco_dict_ = new_aruco_dict;
    params_.dict_type = dict_mappings[aruco_dict_];
  }
  else
  {
    LOG_ERR("Invalid aruco dictionary '" << new_aruco_dict << "' selected!");
    return false;
  }

  param_handler_->handleStaticParameter<bool>("board.use_for_refinement", params_.board.use_for_refinement);

  if (params_.board.use_for_refinement)
  {
    // Improvement note: Rejecting invalid parameters based on range checks would be nice, but is currently (02/2024)
    //                   not implemented within the parameter handler.
    param_handler_->handleStaticParameter<float>("board.checker_size", params_.board.checker_size);
    param_handler_->handleStaticParameter<float>("board.marker_size", params_.board.marker_size);
    param_handler_->handleStaticParameter<int>("board.num_cols", params_.board.num_columns);
    param_handler_->handleStaticParameter<int>("board.num_rows", params_.board.num_rows);
    param_handler_->handleStaticParameter<float>("board.gating_factor", params_.board.gating_factor);
  }

  int single_marker_id_threshold = static_cast<int>(params_.single_marker_id_threshold);
  param_handler_->handleStaticParameter<int>("single_marker_id_threshold", single_marker_id_threshold);
  if (single_marker_id_threshold < 0)
  {
    LOG_ERR("Parameter 'single_marker_id_threshold' has to be >= 0");
    return false;
  }

  params_.single_marker_id_threshold = static_cast<unsigned int>(single_marker_id_threshold);

  param_handler_->handleStaticParameter<double>("min_marker_perimeter_rate", params_.min_marker_perimeter_rate);

  // measurement uncertainty settings
  param_handler_->handleDynamicParameter<float>("meas_uncertainty.reduction_via_board",
                                                uncertainty_reduction_via_board_);
  param_handler_->handleDynamicParameter<float>("meas_uncertainty.position.est_angular_calib_error",
                                                position_uncertainty_.est_angular_calib_error);
  param_handler_->handleDynamicParameter<float>("meas_uncertainty.position.est_xy_calib_error",
                                                position_uncertainty_.est_xy_calib_error);
  param_handler_->handleDynamicParameter<int>("meas_uncertainty.position.est_pixel_error",
                                              position_uncertainty_.est_pixel_error);

  param_handler_->handleDynamicParameter<float>("meas_uncertainty.orientation.base",
                                                orientation_uncertainty_.base_uncertainty);
  param_handler_->handleDynamicParameter<float>("meas_uncertainty.orientation.distance_scaling",
                                                orientation_uncertainty_.uncertainty_scaling_distance);
  param_handler_->handleDynamicParameter<float>("meas_uncertainty.orientation.angle_scaling",
                                                orientation_uncertainty_.uncertainty_scaling_angle);

  // together with the log level these are the only parameters which may be changed during runtime
  param_handler_->handleDynamicParameter("debug.show_markers", debug_params_->show_markers);
  param_handler_->handleDynamicParameter("debug.show_rejected_markers", debug_params_->show_rejected_markers);
  param_handler_->handleDynamicParameter("debug.show_poses", debug_params_->show_poses);
  param_handler_->handleDynamicParameter("debug.show_marker_gating", debug_params_->show_marker_gating);
  param_handler_->handleDynamicParameter("debug.broadcast_detections_as_transforms",
                                         broadcast_detections_as_transforms_);

  // the param handler supports numbers (1,2,...) and strings (debug, info,...)
  param_handler_->handleLoggerLevelParameter(
      "log_level",
      [&](aduulm_logger::LoggerLevel log_level) { _setLogLevel(log_level); },
      aduulm_logger::LoggerLevel::Warn);

  return true;
}

bool CapsulePoseEstimation::advertiseTopics()
{
  if (use_gpu_)
  {
    LOG_INF("Listening for gpu images (undistorted + rectified) on " << DEFAULT_GPU_IMAGE_TOPIC);

    // determine whether the provided image topic should include a rectification of the image
    const auto remapped_topic =
        this->get_node_base_interface()->resolve_topic_or_service_name(DEFAULT_GPU_IMAGE_TOPIC, false);

    rectified_input_ = (remapped_topic.ends_with("_rect") or remapped_topic.find("_rect/") != std::string::npos);

    gpu_image_sub_ = std::make_unique<aduulm_gpu_shared_mem::interfaces::GpuImageSubscriber>(
        this, DEFAULT_GPU_IMAGE_TOPIC, gpu_shared_mem_services_ns_, this->get_fully_qualified_name());

    gpu_image_sub_->setGpuMatCallback([this](const GpuImageMsg& image_msg, const cv::cuda::GpuMat& image) -> void {
      gpuImageCallback(image_msg, image);
    });
  }
  else
  {
    LOG_INF("Listening for normal images (undistorted + rectified) on " << DEFAULT_CPU_IMAGE_TOPIC);

    rclcpp::SensorDataQoS sensor_qos;
    sensor_qos.get_rmw_qos_profile().depth = 1;
    cpu_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        DEFAULT_CPU_IMAGE_TOPIC, sensor_qos, [this](CpuImageMsg::ConstSharedPtr msg) {
          this->cpuImageCallback(std::move(msg));
        });

    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        DEFAULT_CPU_IMAGE_INFO_TOPIC, 1, [this](CameraInfoMsg::ConstSharedPtr msg) {
          this->cpuCameraInfoCallback(std::move(msg));
        });
  }

  debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(DEFAULT_DEBUG_IMAGE_TOPIC, rclcpp::QoS(1));
  marker_detection_pub_ = this->create_publisher<aduulm_object_interfaces::msg::DetectionList>(
      DEFAULT_MARKER_DETECTION_TOPIC, rclcpp::QoS(1));

  return true;
}

void CapsulePoseEstimation::cpuImageCallback(const CpuImageMsg::ConstSharedPtr img_msg)
{
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    LOG_ERR("unable to convert image msg via cv_bridge: %s" << e.what());
    return;
  }

  if (latest_camera_info_)
  {
    imageCallback(cv_ptr->image, img_msg->header, latest_camera_info_.value(), true);
  }
  else
  {
    LOG_WARN_THROTTLE(5.0, "camera_info is required for marker pose estimation, but hasn't been received yet!");
  }
}

void CapsulePoseEstimation::cpuCameraInfoCallback(const CameraInfoMsg::ConstSharedPtr info_msg)
{
  latest_camera_info_ = *info_msg;
}

void CapsulePoseEstimation::gpuImageCallback(const GpuImageMsg& img_msg, const cv::cuda::GpuMat& gpu_image)
{
  // the toolbox is unfortunately not able to handle GPU images directly
  cv::UMat img;
  gpu_image.download(img);

  imageCallback(img, img_msg.header, img_msg.image_info, rectified_input_);
}

void changeAxesConvention(std::vector<lib::Detection>& detections)
{
  // To simplify further processing, we replace the OpenCV convention for axes with our own one:
  //  - OpenCV: x = marker right,  y = marker top,   z = facing camera
  //  - ours:   x = facing camera, y = Marker right, z = marker top
  for (lib::Detection& det : detections)
  {
    // transform opencv coords
    tf2::Quaternion corr_quat;
    corr_quat.setRPY(0, -M_PI_2, -M_PI_2);
    tf2::Quaternion quat(det.rotation_x, det.rotation_y, det.rotation_z, det.rotation_w);
    quat = quat * corr_quat;
    det.rotation_x = quat.x();
    det.rotation_y = quat.y();
    det.rotation_z = quat.z();
    det.rotation_w = quat.w();
  }
}

template <typename Container>
void CapsulePoseEstimation::imageCallback(const Container& image,
                                          const std_msgs::msg::Header& msg_header,
                                          const sensor_msgs::msg::CameraInfo& camera_info,
                                          bool rectified_input)
{
  cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64FC1);
  cv::Mat distortion_coefficients = cv::Mat::zeros(5, 1, CV_64FC1);

  if (!rectified_input)
  {
    distortion_coefficients = extractDistortionCoefficients(camera_info);
  }

  camera_matrix = extractCameraMatrix(camera_info);

  std::vector<lib::Detection> detections = toolbox_->estimateMarkerPoses(image, camera_matrix, distortion_coefficients);

  // change axes convention from opencv to our system
  changeAxesConvention(detections);

  // publishing of detected markers
  const double focal_length = camera_info.k[0];
  DetectionListMsg detection_msg = convertToROSMessage(detections, msg_header, focal_length);
  marker_detection_pub_->publish(std::move(detection_msg));

  if (broadcast_detections_as_transforms_)
  {
    broadcastDetectionsAsTransforms(detections, msg_header);
  }

  // optional publishing of debug information
  if (debug_params_->any_set() && debug_image_pub_->get_subscription_count() > 0)
  {
    cv::Mat debug_img = toolbox_->getDebugImage();

    if (!debug_img.empty())
    {
      const std::string format = (debug_img.channels() == 3) ? "bgr8" : "mono8";

      CpuImageMsg::SharedPtr msg = cv_bridge::CvImage(msg_header, format, debug_img).toImageMsg();

      debug_image_pub_->publish(*msg);
    }
  }
}

CapsulePoseEstimation::DetectionListMsg
CapsulePoseEstimation::convertToROSMessage(const std::vector<lib::Detection>& detections,
                                           const std_msgs::msg::Header& msg_header,
                                           double focal_length) const
{
  CapsulePoseEstimation::DetectionListMsg msg;
  msg.header.stamp = msg_header.stamp;
  msg.header.frame_id = output_frame_id_;
  msg.meta_info.emplace_back("frame_id:" + msg_header.frame_id);

  if (detections.empty())
  {
    return msg;
  }

  std::optional<geometry_msgs::msg::TransformStamped> sensor_to_output_trafo;
  try
  {
    sensor_to_output_trafo = tf_buffer_.lookupTransform(output_frame_id_, msg_header.frame_id, msg_header.stamp);
  }
  catch (tf2::ExtrapolationException& ex)
  {
    sensor_to_output_trafo = tf_buffer_.lookupTransform(output_frame_id_, msg_header.frame_id, tf2::TimePointZero);
  }
  catch (tf2::TransformException& ex)
  {
    LOG_WARN_THROTTLE(1.0,
                      "TransformException for transform from " << msg_header.frame_id << " to " << output_frame_id_);
    return msg;
  }

  for (const lib::Detection& detection : detections)
  {
    aduulm_object_interfaces::msg::Detection detection_msg;
    detection_msg.id = detection.id;
    detection_msg.existence_prob = 1.0;
    detection_msg.stamp = msg_header.stamp;

    geometry_msgs::msg::Pose pose_camera_frame = poseFromDetection(detection);

    tf2::Quaternion pose_quat;
    tf2::fromMsg(pose_camera_frame.orientation, pose_quat);

    LOG_DEB("marker id " << detection.id);

    // Create covariances
    const double dist_to_sensor = cv::norm(cv::Vec3d(detection.x, detection.y, detection.z));
    const double angle_to_camera = std::atan2(-detection.x, detection.z);  // det is in camera frame

    // create vectors pointing from marker to camera and one at marker center
    tf2::Vector3 vec_marker_to_camera(1.0 * sin(angle_to_camera), 0.0, -1.0 * cos(angle_to_camera));
    tf2::Vector3 vec_marker = tf2::quatRotate(pose_quat, tf2::Vector3(1.0, 0.0, 0.0));

    // eliminate height (y-dimension)
    vec_marker.setY(0.0);
    vec_marker_to_camera.setY(0.0);

    LOG_DEB("vec_marker_to_camera " << vec_marker_to_camera.x() << ", " << vec_marker_to_camera.y() << ", "
                                    << vec_marker_to_camera.z());
    LOG_DEB("vec_marker " << vec_marker.x() << ", " << vec_marker.y() << ", " << vec_marker.z());

    const double dot = vec_marker.x() * vec_marker_to_camera.x() + vec_marker.z() * vec_marker_to_camera.z();
    const double magnitudes = vec_marker.length() * vec_marker_to_camera.length();
    const double abs_marker_angle_to_camera_ray = acos(dot / magnitudes);
    LOG_DEB("marker_angle_to_camera_ray " << abs_marker_angle_to_camera_ray);

    const std::string info = "marker_angle_to_camera_ray:" + std::to_string(abs_marker_angle_to_camera_ray);
    detection_msg.meta_info.emplace_back(info.c_str());
    const std::array<double, 36> pos_cov = getPositionCovariance(
        abs_marker_angle_to_camera_ray, dist_to_sensor, angle_to_camera, detection.refined_by_board, focal_length);
    const double yaw_variance = getYawStddev(dist_to_sensor, angle_to_camera, detection.refined_by_board);

    // transform covariances to output frame
    const tf2::Transform sensor2output_trafo(tf2::Quaternion(sensor_to_output_trafo.value().transform.rotation.x,
                                                             sensor_to_output_trafo.value().transform.rotation.y,
                                                             sensor_to_output_trafo.value().transform.rotation.z,
                                                             sensor_to_output_trafo.value().transform.rotation.w),
                                             tf2::Vector3(sensor_to_output_trafo.value().transform.translation.x,
                                                          sensor_to_output_trafo.value().transform.translation.y,
                                                          sensor_to_output_trafo.value().transform.translation.z));
    std::array<double, 36> output_frame_pos_cov = tf2::transformCovariance(pos_cov, sensor2output_trafo);

    // transform with angle in camera to make covariance be aligned with the ray away from camera origin
    tf2::Quaternion new_quat;
    new_quat.setRPY(0, 0, angle_to_camera);
    tf2::Vector3 zero_trafo(0, 0, 0);
    const tf2::Transform output_to_camera_angle(new_quat, zero_trafo);
    std::array<double, 36> camera_angle_pos_cov =
        tf2::transformCovariance(output_frame_pos_cov, output_to_camera_angle);

    // remap into 2D matrix
    Eigen::Matrix<double, 6, 6> cov_mat =
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(camera_angle_pos_cov.data());

    // convert into aduulm_interfaces representation
    using MsgStateType = aduulm_object_types_lib::CP3D_Y;
    assert(aduulm_object_types_lib::ATTR_IDS.find(MsgStateType::Type) != aduulm_object_types_lib::ATTR_IDS.end());
    auto object_state = aduulm_object_types_lib::wrap<MsgStateType>(detection_msg.state);
    object_state.init();

    // transform pose
    geometry_msgs::msg::Pose pose_output_trafo;
    tf2::doTransform(pose_camera_frame, pose_output_trafo, sensor_to_output_trafo.value());

    // The output state is limited to what the tracking can currently handle (e.g. 3D position + yaw angle).
    object_state.attr(MsgStateType::X) = pose_output_trafo.position.x;
    object_state.attr(MsgStateType::Y) = pose_output_trafo.position.y;
    object_state.attr(MsgStateType::Z) = pose_output_trafo.position.z;
    tf2::fromMsg(pose_output_trafo.orientation, pose_quat);
    object_state.attr(MsgStateType::YAW) = tf2::getYaw(pose_quat);

    object_state.var(MsgStateType::X) = cov_mat(0, 0);
    object_state.var(MsgStateType::Y) = cov_mat(1, 1);
    object_state.var(MsgStateType::Z) = cov_mat(1, 1);
    object_state.cov(MsgStateType::X, MsgStateType::Y) = cov_mat(0, 1);
    object_state.cov(MsgStateType::X, MsgStateType::Z) = cov_mat(0, 2);
    object_state.cov(MsgStateType::Y, MsgStateType::X) = cov_mat(1, 0);
    object_state.cov(MsgStateType::Y, MsgStateType::Z) = cov_mat(1, 2);
    object_state.cov(MsgStateType::Z, MsgStateType::X) = cov_mat(2, 0);
    object_state.cov(MsgStateType::Z, MsgStateType::Y) = cov_mat(2, 1);
    object_state.var(MsgStateType::YAW) = std::pow(yaw_variance, 2);

    detection_msg.state.reference_point.value = aduulm_object_interfaces::msg::ReferencePoint::GEOMETRIC_CENTER;
    aduulm_object_interfaces::msg::ClassificationProbability class_prob;
    class_prob.probability = 1.0;
    class_prob.type = aduulm_object_interfaces::msg::ClassificationType::TARGET;
    detection_msg.classification.classification_prob.push_back(class_prob);

    if (detection.refined_by_board)
    {
      detection_msg.meta_info.emplace_back("refined");
    }

    msg.detections.push_back(detection_msg);
  }

  return msg;
}

/**
 * Get variance of yaw in up-right coordinate system
 * @param dist_to_sensor
 * @param angle_to_camera
 * @param is_refined
 * @return
 */
double CapsulePoseEstimation::getYawStddev(double dist_to_sensor, double angle_to_camera, bool is_refined) const
{
  double orientation_stddev = orientation_uncertainty_.base_uncertainty +
                              orientation_uncertainty_.uncertainty_scaling_distance * dist_to_sensor +
                              orientation_uncertainty_.uncertainty_scaling_angle * std::abs(angle_to_camera);

  if (is_refined)
  {
    orientation_stddev *= uncertainty_reduction_via_board_;
  }

  // clamp orientation stddev to avoid invalid values (i.e. > 360°)
  orientation_stddev = std::min(orientation_stddev, MAX_ORIENTATION_STDDEV_RAD);
  return orientation_stddev;
}

/**
 * Get 3D covariance in camera frame
 * @param dist_to_sensor
 * @param angle_to_camera
 * @param is_refined
 * @return
 */
std::array<double, 36> CapsulePoseEstimation::getPositionCovariance(double marker_angle,
                                                                    double dist_to_sensor,
                                                                    double angle_to_camera,
                                                                    bool is_refined,
                                                                    double focal_length) const
{
  double position_stddev_xy = getXYStddev(dist_to_sensor, focal_length, position_uncertainty_);
  double position_stddev_z =
      getDepthStddev(marker_angle, dist_to_sensor, params_.marker_size, focal_length, position_uncertainty_);

  if (is_refined)
  {
    position_stddev_xy *= uncertainty_reduction_via_board_;
    position_stddev_z *= uncertainty_reduction_via_board_;
  }

  Eigen::Matrix<double, 6, 6> cov_mat = Eigen::Matrix<double, 6, 6>::Zero();
  cov_mat.diagonal() << std::pow(position_stddev_xy, 2), std::pow(position_stddev_xy, 2),
      std::pow(position_stddev_z, 2), 0.0, 0.0, 0.0;

  // copy into flat array
  std::array<double, 36> cov_array;
  for (int i = 0; i < cov_mat.rows(); ++i)
  {
    for (int j = 0; j < cov_mat.cols(); ++j)
    {
      cov_array[i * cov_mat.cols() + j] = cov_mat(i, j);
    }
  }

  return cov_array;
}

void CapsulePoseEstimation::broadcastDetectionsAsTransforms(const std::vector<lib::Detection>& detections,
                                                            const std_msgs::msg::Header& msg_header)
{
  // Note: for debugging purposes we publish two transformations:
  //       1) a transformation matching the debug image generated using opencv (denoted by _opencv)
  //       2) a transformation using our coordinate frame conventions which is used for determining the output
  //          yaw angle (x = facing camera, y = marker right, z = marker top)

  for (const auto& detection : detections)
  {
    // publish normal trafo
    geometry_msgs::msg::TransformStamped output_trafo;
    output_trafo.header.frame_id = msg_header.frame_id;
    output_trafo.header.stamp = msg_header.stamp;
    output_trafo.child_frame_id = "marker_" + std::to_string(detection.id);
    output_trafo.transform.translation.x = detection.x;
    output_trafo.transform.translation.y = detection.y;
    output_trafo.transform.translation.z = detection.z;
    output_trafo.transform.rotation.x = detection.rotation_x;
    output_trafo.transform.rotation.y = detection.rotation_y;
    output_trafo.transform.rotation.z = detection.rotation_z;
    output_trafo.transform.rotation.w = detection.rotation_w;
    tf_broadcaster_->sendTransform(output_trafo);

    // rotate back to opencv trafo
    geometry_msgs::msg::TransformStamped opencv_trafo;
    opencv_trafo.header.stamp = msg_header.stamp;
    opencv_trafo.header.frame_id = "marker_" + std::to_string(detection.id);
    opencv_trafo.child_frame_id = "marker_" + std::to_string(detection.id) + "_opencv";
    tf2::Quaternion quat;
    quat.setRPY(0, -M_PI_2, -M_PI_2);
    quat = quat.inverse();
    opencv_trafo.transform.rotation.x = quat.x();
    opencv_trafo.transform.rotation.y = quat.y();
    opencv_trafo.transform.rotation.z = quat.z();
    opencv_trafo.transform.rotation.w = quat.w();

    tf_broadcaster_->sendTransform(opencv_trafo);
  }
}

cv::Mat extractDistortionCoefficients(const sensor_msgs::msg::CameraInfo& msg)
{
  cv::Mat distortion_coefficients(static_cast<int>(msg.d.size()), 1, CV_64FC1);

  for (auto i = 0U; i < msg.d.size(); i++)
  {
    distortion_coefficients.at<double>(static_cast<int>(i), 0) = msg.d.at(i);
  }

  return distortion_coefficients;
}

cv::Mat extractCameraMatrix(const sensor_msgs::msg::CameraInfo& msg)
{
  //     [fx  0 cx Tx]
  // P = [ 0 fy cy Ty] (provided as row major array)
  //     [ 0  0  1 0 ]
  cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64FC1);

  camera_matrix.at<double>(0, 0) = msg.p.at(0);
  camera_matrix.at<double>(0, 2) = msg.p.at(2);
  camera_matrix.at<double>(1, 1) = msg.p.at(5);
  camera_matrix.at<double>(1, 2) = msg.p.at(6);

  return camera_matrix;
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

// we only need overloads for cv::UMat and cv::Mat, thus we can explicitly instantiate the templates
// (thereby allowing the implementation to stay within the .cpp file)
template void CapsulePoseEstimation::imageCallback<cv::UMat>(const cv::UMat& image,
                                                             const std_msgs::msg::Header& msg_header,
                                                             const sensor_msgs::msg::CameraInfo& camera_info,
                                                             bool rectified_input = true);
template void CapsulePoseEstimation::imageCallback<cv::Mat>(const cv::Mat& image,
                                                            const std_msgs::msg::Header& msg_header,
                                                            const sensor_msgs::msg::CameraInfo& camera_info,
                                                            bool rectified_input = true);

DEFINE_LOGGER_CLASS_INTERFACE_IMPLEMENTATION(CapsulePoseEstimation)
}  // namespace capsule_pose_estimation

int main(int argc, char** argv)
{
  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  rclcpp::init(argc, argv);

  auto node = std::make_shared<capsule_pose_estimation::CapsulePoseEstimation>("capsule_pose_estimation");

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