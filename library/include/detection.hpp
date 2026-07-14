#ifndef CAPSULE_POSE_ESTIMATION_DETECTION_HPP
#define CAPSULE_POSE_ESTIMATION_DETECTION_HPP

#include <Eigen/Core>

namespace capsule_pose_estimation::lib
{
struct Detection
{
  double x = 0;
  double y = 0;
  double z = 0;

  float rotation_x;
  float rotation_y;
  float rotation_z;
  float rotation_w;

  Eigen::Matrix<double, 2, 2> pos_cov;

  /// ID of the (main) aruco marker.
  unsigned int id = 0;
  /// effective width depending on the angle of the marker to the camera ray towards it
  double marker_angle_to_camera_ray = 0.0;

  /// If true, the pose of the single marker has been refined using a charuco board / lidar. Otherwise the pose is
  /// solely estimated for the single marker.
  bool refined_by_board = false;
  bool refined_by_lidar = false;
};
}  // namespace capsule_pose_estimation::lib

#endif  // CAPSULE_POSE_ESTIMATION_DETECTION_HPP
