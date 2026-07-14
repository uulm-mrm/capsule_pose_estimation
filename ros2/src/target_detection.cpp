/**
 * @file capsule_pose_estimation.cpp
 *
 * @brief Node for the pose estimation of U-Shift II capsules
 *
 * @author Oliver Schumann
 * Contact: oliver.schumann@uni-ulm.de
 *
 */
#include "target_detection.hpp"

#include <map>
#include <utility>

#include <aduulm_object_types_lib/object_access.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/utils.h>  // has to be included after tf2_geometry_msgs!

#include "logger_setup.hpp"

DEFINE_LOGGER_VARIABLES

namespace capsule_pose_estimation
{
static constexpr auto DEFAULT_MARKER_DETECTION_TOPIC = "refined_marker_detections";
static constexpr auto DEFAULT_TARGET_DETECTION_TOPIC = "target_detections";

TargetDetection::TargetDetection(const std::string& name)
  : rclcpp_lifecycle::LifecycleNode{ name,
                                     rclcpp::NodeOptions()
                                         .allow_undeclared_parameters(true)
                                         .automatically_declare_parameters_from_overrides(true) }
  , param_handler_(std::make_unique<ParameterHandler>(this->get_node_parameters_interface(), this))
{
  initLogger();
}

TargetDetection::CallbackReturn TargetDetection::on_configure(const rclcpp_lifecycle::State& state)
{
  handleParameter();

  advertiseTopics();

  targets_ = std::make_unique<Targets>(target_config_file_);

  LOG_INF("done configuring");

  return CallbackReturn::SUCCESS;
}

TargetDetection::CallbackReturn TargetDetection::on_activate(const rclcpp_lifecycle::State& state)
{
  LOG_INF("activating...");
  target_detection_pub_->on_activate();
  LOG_INF("done activating");
  return CallbackReturn::SUCCESS;
}

TargetDetection::CallbackReturn TargetDetection::on_deactivate(const rclcpp_lifecycle::State& state)
{
  target_detection_pub_->on_deactivate();
  return CallbackReturn::SUCCESS;
}

TargetDetection::CallbackReturn TargetDetection::on_cleanup(const rclcpp_lifecycle::State& state)
{
  return CallbackReturn::SUCCESS;
}

TargetDetection::CallbackReturn TargetDetection::on_shutdown(const rclcpp_lifecycle::State& state)
{
  return CallbackReturn::SUCCESS;
}

void TargetDetection::initLogger()
{
  _initLogger();
  _setStreamName("TargetDetection");
  _setLogLevel(aduulm_logger::LoggerLevel::Warn);
  LOGGER_ADD_SUBLOGGER_LIBRARY(capsule_pose_estimation::lib);
}

bool TargetDetection::handleParameter()
{
  param_handler_->handleStaticParameter<std::string>("target_config_file", target_config_file_);
  param_handler_->handleDynamicParameter<double>(
      "maximum_detection_height", maximum_detection_height_, maximum_detection_height_);

  // the param handler supports numbers (1,2,...) and strings (debug, info,...)
  param_handler_->handleLoggerLevelParameter(
      "log_level",
      [&](aduulm_logger::LoggerLevel log_level) { _setLogLevel(log_level); },
      aduulm_logger::LoggerLevel::Warn);

  return true;
}

bool TargetDetection::advertiseTopics()
{
  marker_sub_ = this->create_subscription<aduulm_object_interfaces::msg::DetectionList>(
      DEFAULT_MARKER_DETECTION_TOPIC, 1, [this](aduulm_object_interfaces::msg::DetectionList::ConstSharedPtr msg) {
        this->markerCallback(msg);
      });

  target_detection_pub_ =
      this->create_publisher<aduulm_object_interfaces::msg::DetectionList>(DEFAULT_TARGET_DETECTION_TOPIC, 1);

  return true;
}

void TargetDetection::markerCallback(aduulm_object_interfaces::msg::DetectionList::ConstSharedPtr msg)
{
  // transform to target detections and publish them
  // Note: Depending on the use case it is necessary to track targets (capsules) instead of individual markers.
  //       A capsule can be equipped with multiple markers, the individual positions within the capsule coordinate frame
  //       are stored within a config file.
  //       Since the employed tracking approach handles assumes that an object generates at most one detection per
  //       sensor, we filter the detection within the conversion step and keep only one per capsule (the one with
  //       the lowest uncertainty)
  const DetectionListMsg target_detections_msg = transformToTargetDetections(*msg);

  target_detection_pub_->publish(target_detections_msg);
}

TargetDetection::DetectionListMsg
TargetDetection::transformToTargetDetections(const aduulm_object_interfaces::msg::DetectionList& msg,
                                             const bool allow_duplicates) const
{
  std::set<unsigned int> processed_targets;

  DetectionListMsg target_msg_list;
  target_msg_list.header = msg.header;
  target_msg_list.meta_info = msg.meta_info;

  for (const auto& detection_msg : msg.detections)
  {
    // search for detected marker in targets
    std::optional<MarkerConfig> marker_config = std::nullopt;
    std::optional<unsigned int> target_id = std::nullopt;
    for (const TargetConfig& target : targets_->target_configs_)
    {
      // checking if the target config contains the marker
      const auto it = target.marker_configs_.find(static_cast<int>(detection_msg.id));
      if (it != target.marker_configs_.end())
      {
        LOG_INF_THROTTLE(1.0,
                         "Marker with id " << detection_msg.id << " found on target of type " << target.type_
                                           << " with name " << target.name_ << " and id " << target.id_);
        marker_config = it->second;
        target_id = target.id_;
        break;
      }
    }

    if (not marker_config.has_value())
    {
      LOG_INF_THROTTLE(1.0, "Marker with id " << detection_msg.id << " not found in target config");
      continue;
    }

    // copy detection message (state is later overridden based on transformed estimate)
    aduulm_object_interfaces::msg::Detection target_msg = detection_msg;
    target_msg.id = target_id.value();

    using MsgStateType = aduulm_object_types_lib::CP3D_Y;
    assert(aduulm_object_types_lib::ATTR_IDS.find(MsgStateType::Type) != aduulm_object_types_lib::ATTR_IDS.end());
    auto object_state = aduulm_object_types_lib::wrap<MsgStateType>(target_msg.state);

    // sensor -> marker
    const Eigen::Quaterniond quat(Eigen::AngleAxisd(object_state.attr(MsgStateType::YAW), Eigen::Vector3d::UnitZ()));
    const Eigen::Matrix3d rotation(quat.toRotationMatrix());
    Eigen::Matrix4d sensor2marker;
    sensor2marker.topLeftCorner<3, 3>() = rotation;
    sensor2marker(0, 3) = object_state.attr(MsgStateType::X);
    sensor2marker(1, 3) = object_state.attr(MsgStateType::Y);
    sensor2marker(2, 3) = object_state.attr(MsgStateType::Z);
    sensor2marker(3, 3) = 1.0;

    // compute final trafo
    const Eigen::Matrix4d full_trafo = sensor2marker * marker_config->marker2target_;

    object_state.attr(MsgStateType::X) = full_trafo(0, 3);
    object_state.attr(MsgStateType::Y) = full_trafo(1, 3);
    object_state.attr(MsgStateType::Z) = full_trafo(2, 3);

    const Eigen::Matrix3d det_rotation = full_trafo.topLeftCorner<3, 3>();
    const Eigen::Quaterniond det_quat(det_rotation);
    tf2::Quaternion det_quat_tf(det_quat.x(), det_quat.y(), det_quat.z(), det_quat.w());
    object_state.attr(MsgStateType::YAW) = tf2::getYaw(det_quat_tf);

    if (double height = object_state.attr(MsgStateType::Z);
        height > maximum_detection_height_ or height < -maximum_detection_height_)
    {
      LOG_ERR("Flying detection ignored with height = " << height);
      continue;
    }

    if (not allow_duplicates and processed_targets.contains(target_id.value()))
    {
      // goal: keep only the detection with the lower (positional) uncertainty (determined via trace of covariance
      //       matrix)
      // TODO (Schumann) refactor this!
      std::size_t prev_target_idx = 0;
      for (; prev_target_idx < target_msg_list.detections.size(); prev_target_idx++)
      {
        if (target_msg_list.detections[prev_target_idx].id == target_id.value())
        {
          break;
        }
      }
      assert(prev_target_idx != target_msg_list.detections.size());

      auto old_object_state =
          aduulm_object_types_lib::wrap<MsgStateType>(target_msg_list.detections[prev_target_idx].state);

      // trace computation
      const auto trace_new_detection =
          object_state.var(MsgStateType::X) + object_state.var(MsgStateType::Y) + object_state.var(MsgStateType::Z);

      const auto trace_old_detection = old_object_state.var(MsgStateType::X) + old_object_state.var(MsgStateType::Y) +
                                       old_object_state.var(MsgStateType::Z);

      // override previous target message in case the new one has a lower positional uncertainty
      if (trace_new_detection < trace_old_detection)
      {
        target_msg_list.detections[prev_target_idx] = target_msg;
      }
    }
    else
    {
      processed_targets.emplace(target_id.value());
      target_msg_list.detections.emplace_back(target_msg);
    }
  }

  return target_msg_list;
}

DEFINE_LOGGER_CLASS_INTERFACE_IMPLEMENTATION(TargetDetection)
}  // namespace capsule_pose_estimation

int main(int argc, char** argv)
{
  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  rclcpp::init(argc, argv);

  auto node = std::make_shared<capsule_pose_estimation::TargetDetection>("capsule_pose_estimation");

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