/**
 * @file camera_detection.hpp
 *
 * @brief Node for the pose estimation of U-Shift II capsules
 *
 * @author Oliver Schumann
 * Contact: oliver.schumann@uni-ulm.de
 *
 */

#ifndef CAPSULE_POSE_ESTIMATION_HPP
#define CAPSULE_POSE_ESTIMATION_HPP

#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <aduulm_logger/aduulm_logger.hpp>
#include <aduulm_tools/parameter_utils/parameter_handler.hpp>
#include <aduulm_gpu_shared_mem/interfaces/gpu_shared_mem_handler.hpp>
#include <aduulm_gpu_shared_mem/interfaces/gpu_image_subscriber.hpp>
#include <aduulm_object_interfaces/msg/detection_list.hpp>

#include "aruco_toolbox.hpp"

namespace capsule_pose_estimation
{
struct PositionMeasurementUncertaintyParams
{
  float est_angular_calib_error = 0.00872665;  // 0.5 deg
  float est_xy_calib_error = 0.01;
  int est_pixel_error = 3;
};

struct OrientationMeasurementUncertaintyParams
{
  // Note:
  // All parameters are representing standard deviations!
  float base_uncertainty = 0.05F;
  float uncertainty_scaling_distance = 0.05F;
  float uncertainty_scaling_angle = 0.25F;
};

double getXYStddev(double distance, double focal_length, const PositionMeasurementUncertaintyParams& params)
{
  const double by_pixel_error = params.est_pixel_error / focal_length * distance;
  const double by_calibration_error = distance * tan(params.est_angular_calib_error) + params.est_xy_calib_error;
  return by_pixel_error + by_calibration_error;
}

double getDepthStddev(double marker_angle,
                      double distance,
                      double marker_size,
                      double focal_length,
                      const PositionMeasurementUncertaintyParams& params)
{
  const double marker_angle_reduction_factor = cos(abs(marker_angle));
  const double marker_width_px = marker_size * focal_length / distance * marker_angle_reduction_factor;
  const double by_pixel_error = abs(marker_width_px / (marker_width_px + params.est_pixel_error) - 1) * distance;
  const double by_calibration_error = params.est_xy_calib_error;
  return by_pixel_error + by_calibration_error;
}

class CapsulePoseEstimation : public rclcpp_lifecycle::LifecycleNode
{
public:
  using ParameterHandler = aduulm_tools::parameter_utils::ParameterHandler;
  using CallbackReturn = rclcpp_lifecycle::LifecycleNode::CallbackReturn;
  using CpuImageMsg = sensor_msgs::msg::Image;
  using CameraInfoMsg = sensor_msgs::msg::CameraInfo;
  using GpuImageMsg = aduulm_gpu_shared_mem::interfaces::GpuImageSubscriber::GpuImageMsg;
  using GpuImageSubscriber = aduulm_gpu_shared_mem::interfaces::GpuImageSubscriber;
  using DetectionListMsg = aduulm_object_interfaces::msg::DetectionList;

  // ros helper defs
  template <class T>
  using PublisherPtr = typename rclcpp_lifecycle::LifecyclePublisher<T>::SharedPtr;
  template <class T>
  using SubscriptionPtr = typename rclcpp::Subscription<T>::SharedPtr;

  DEFINE_LOGGER_CLASS_INTERFACE_HEADER

  /**
   * @brief Constructor.
   * @param name Node name.
   */
  explicit CapsulePoseEstimation(const std::string& name);

  /**
   * @brief virtual dtor is necessary
   */
  ~CapsulePoseEstimation() override = default;

  /**
   * lifecycle function for on_configure
   * @return callback return status
   */
  CallbackReturn on_configure(const rclcpp_lifecycle::State& /*previous_state*/) override;

  /**
   * lifecycle function for on_activate
   * @return callback return status
   */
  CallbackReturn on_activate(const rclcpp_lifecycle::State& /*previous_state*/) override;

  /**
   * lifecycle function for on_deactivate
   * @return callback return status
   */
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& /*previous_state*/) override;

  /**
   * lifecycle function for on_cleanup
   * @return callback return status
   */
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State& /*previous_state*/) override;

  /**
   * lifecycle function for on_shutdown
   * @return callback return status
   */
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State& /*previous_state*/) override;

protected:
  void initLogger();

  bool handleParameter();
  bool advertiseTopics();

  void cpuImageCallback(CpuImageMsg::ConstSharedPtr img_msg);
  void cpuCameraInfoCallback(CameraInfoMsg::ConstSharedPtr info_msg);

  void gpuImageCallback(const GpuImageMsg& img_msg, const cv::cuda::GpuMat& gpu_image);
  template <typename Container>
  void imageCallback(const Container& image,
                     const std_msgs::msg::Header& msg_header,
                     const sensor_msgs::msg::CameraInfo& camera_info,
                     bool rectified_input = true);

  [[nodiscard]] double getYawStddev(double dist_to_sensor, double angle_to_camera, bool is_refined) const;
  [[nodiscard]] std::array<double, 36> getPositionCovariance(double marker_angle,
                                                             double dist_to_sensor,
                                                             double angle_to_camera,
                                                             bool is_refined,
                                                             double focal_length) const;

  [[nodiscard]] DetectionListMsg convertToROSMessage(const std::vector<lib::Detection>& detections,
                                                     const std_msgs::msg::Header& msg_header,
                                                     double focal_length) const;

  void broadcastDetectionsAsTransforms(const std::vector<lib::Detection>& detections,
                                       const std_msgs::msg::Header& msg_header);

  bool use_gpu_ = true;
  bool rectified_input_ = true;
  bool broadcast_detections_as_transforms_ = false;
  std::string gpu_shared_mem_services_ns_;
  std::string aruco_dict_ = "4x4_50";
  std::string output_frame_id_;

  PositionMeasurementUncertaintyParams position_uncertainty_;
  OrientationMeasurementUncertaintyParams orientation_uncertainty_;
  // Scaling factor applied to the uncertainty if the detection has been refined using the Charuco board
  float uncertainty_reduction_via_board_ = 0.25;

  lib::Parameters params_;
  std::shared_ptr<lib::DebuggingParameters> debug_params_;
  std::unique_ptr<lib::ArucoToolbox> toolbox_;

  // Publisher/Subscription
  // TODO(@authaler,@schumann): should be replaced with image transport once that supports lifecycle nodes
  SubscriptionPtr<sensor_msgs::msg::Image> cpu_image_sub_{ nullptr };
  SubscriptionPtr<sensor_msgs::msg::CameraInfo> camera_info_sub_{ nullptr };
  std::unique_ptr<GpuImageSubscriber> gpu_image_sub_{ nullptr };
  std::unique_ptr<ParameterHandler> param_handler_{ nullptr };
  PublisherPtr<sensor_msgs::msg::Image> debug_image_pub_{ nullptr };
  PublisherPtr<DetectionListMsg> marker_detection_pub_{ nullptr };

  std::optional<CameraInfoMsg> latest_camera_info_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  // mainly intended for debugging purposes
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace capsule_pose_estimation

#endif  // CAPSULE_POSE_ESTIMATION_HPP
