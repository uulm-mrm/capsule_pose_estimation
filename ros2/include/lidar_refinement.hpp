#ifndef LIDAR_REFINEMENT_HPP
#define LIDAR_REFINEMENT_HPP

#include <aduulm_logger/aduulm_logger.hpp>
#include <aduulm_tools/parameter_utils/parameter_handler.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

// interfaces for markers
#include <aduulm_object_interfaces/msg/detection_list.hpp>

// interfaces for points
#include <aduulm_measurement_types_lib/point_cloud_types.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "lidar_refinement_toolbox.hpp"

namespace capsule_pose_estimation
{

class LidarMarkerRefinement : public rclcpp_lifecycle::LifecycleNode
{
public:
  using ParameterHandler = aduulm_tools::parameter_utils::ParameterHandler;
  using CallbackReturn = rclcpp_lifecycle::LifecycleNode::CallbackReturn;
  using DetectionListMsg = aduulm_object_interfaces::msg::DetectionList;

  // ros helper defs
  template <class T>
  using PublisherPtr = typename rclcpp_lifecycle::LifecyclePublisher<T>::SharedPtr;
  template <class T>
  using SubscriptionPtr = typename rclcpp::Subscription<T>::SharedPtr;

  using AduulmPoint = aduulm_measurement_types_lib::AduulmPoint;

  DEFINE_LOGGER_CLASS_INTERFACE_HEADER

  /**
   * @brief Constructor.
   * @param name Node name.
   */
  explicit LidarMarkerRefinement(const std::string& name);

  /**
   * @brief virtual dtor is necessary
   */
  ~LidarMarkerRefinement() override = default;

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
  // helper functions
  void initLogger();
  bool handleParameter();
  bool advertiseTopics();

  // Callbacks for subscribers
  void lidarCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void markerCallback(aduulm_object_interfaces::msg::DetectionList::ConstSharedPtr msg);

  void publishRefinedMarkers(aduulm_object_interfaces::msg::DetectionList::ConstSharedPtr msg,
                             const std::vector<lib::Detection>& refined_markers,
                             const std::string& lidar_frame_id);

  // Subscriber
  SubscriptionPtr<aduulm_object_interfaces::msg::DetectionList> marker_sub_{ nullptr };
  SubscriptionPtr<sensor_msgs::msg::PointCloud2> lidar_sub_;

  // Publisher
  PublisherPtr<aduulm_object_interfaces::msg::DetectionList> refined_marker_pub_{ nullptr };
  PublisherPtr<sensor_msgs::msg::PointCloud2> debug_cloud_publisher_{ nullptr };

  std::unique_ptr<ParameterHandler> param_handler_{ nullptr };

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::shared_ptr<pcl::PointCloud<AduulmPoint>> last_pcl_;
  std::string lidar_frame_id_;

  // toolbox for actual work
  std::unique_ptr<lib::lidar::RefinementToolbox> refinement_toolbox_;
  std::shared_ptr<lib::lidar::RefinementParams> params_;
};

}  // namespace capsule_pose_estimation

#endif  // LIDAR_REFINEMENT_HPP
