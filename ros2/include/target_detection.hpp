/**
 * @file target_detection.hpp
 *
 * @brief Node for the pose estimation of U-Shift II capsules
 *
 * @author Oliver Schumann
 * Contact: oliver.schumann@uni-ulm.de
 *
 */

#ifndef TARGET_DETECTION_HPP
#define TARGET_DETECTION_HPP

#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <aduulm_logger/aduulm_logger.hpp>
#include <aduulm_tools/parameter_utils/parameter_handler.hpp>
#include <aduulm_object_interfaces/msg/detection_list.hpp>

#include <yaml-cpp/yaml.h>

#include "target_config.hpp"

namespace capsule_pose_estimation
{

class TargetDetection : public rclcpp_lifecycle::LifecycleNode
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

  DEFINE_LOGGER_CLASS_INTERFACE_HEADER

  /**
   * @brief Constructor.
   * @param name Node name.
   */
  explicit TargetDetection(const std::string& name);

  /**
   * @brief virtual dtor is necessary
   */
  ~TargetDetection() override = default;

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

  void markerCallback(aduulm_object_interfaces::msg::DetectionList::ConstSharedPtr msg);

  [[nodiscard]] DetectionListMsg transformToTargetDetections(const aduulm_object_interfaces::msg::DetectionList& msg,
                                                             bool allow_duplicates = false) const;

  std::string target_config_file_;

  double maximum_detection_height_ = 2.0;

  // Publisher/Subscription
  // TODO(@authaler,@schumann): should be replaced with image transport once that supports lifecycle nodes
  std::unique_ptr<ParameterHandler> param_handler_{ nullptr };

  SubscriptionPtr<aduulm_object_interfaces::msg::DetectionList> marker_sub_{ nullptr };
  PublisherPtr<DetectionListMsg> target_detection_pub_{ nullptr };

  std::unique_ptr<Targets> targets_;
};

}  // namespace capsule_pose_estimation

#endif  // TARGET_DETECTION_HPP
