#include "lidar_refinement_toolbox.hpp"
#include "logger_setup.hpp"

#define PCL_NO_PRECOMPILE  // needed since we use custom point types
#include <pcl/filters/crop_box.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <Eigen/Geometry>

namespace capsule_pose_estimation::lib::lidar
{

static Eigen::Quaterniond extractOrientation(const pcl::ModelCoefficients& coefficients);
static Eigen::Vector3d calculateRayPlaneIntersection(const pcl::ModelCoefficients& coefficients,
                                                     const Pos3D& camera_position,
                                                     const lib::Detection& marker);

RefinementToolbox::RefinementToolbox(std::shared_ptr<RefinementParams> params) : params_(std::move(params))
{
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
}

std::vector<Detection> RefinementToolbox::refineMarkers(const std::vector<Detection>& markers,
                                                        const PointCloudPtr& points,
                                                        const std::string& camera_frame,
                                                        std::optional<Pos3D> camera_position)
{
  std::vector<Detection> refined_markers;
  refined_markers.reserve(markers.size());

  if (params_->prepare_debug_pcl)
  {
    debug_pcl_.clear();
  }

  pcl::SACSegmentation<AduulmPoint> seg;
  seg.setOptimizeCoefficients(params_->ransac_params.optimize_coefficients);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(params_->ransac_params.method_type);
  seg.setDistanceThreshold(params_->ransac_params.ransac_threshold);

  for (const auto& marker : markers)
  {
    LOG_DEB("Refining marker " << marker.id);
    if (marker.refined_by_board and not params_->refine_boards)
    {
      LOG_DEB("camera_center already refined the marker, forwarding camera-only detection");
      refined_markers.push_back(marker);
      continue;
    }

    const double dist = std::sqrt(std::pow(marker.x, 2) + std::pow(marker.y, 2));
    if (dist < params_->min_distance)
    {
      refined_markers.push_back(marker);
      continue;
    }

    const PointCloudPtr relevant_points = extractRelevantPoints(points, marker);

    if (relevant_points->empty())
    {
      LOG_DEB("  --> there aren't any lidar points near the marker, suppressing detection");
      continue;
    }

    if (params_->prepare_debug_pcl)
    {
      debug_pcl_ += *relevant_points;
    }

    // estimate plane via RANSAC
    pcl::ModelCoefficients coefficients;
    pcl::PointIndices inliers;

    if (relevant_points->size() < 3)
    {
      LOG_DEB("too few points found, forwarding camera-only detection");
      refined_markers.push_back(marker);
      continue;
    }

    seg.setInputCloud(relevant_points);
    seg.segment(inliers, coefficients);

    if (inliers.indices.empty())
    {
      LOG_DEB("  --> RANSAC failed, forwarding camera-only detection");
      refined_markers.push_back(marker);
      continue;
    }

    lib::Detection refined_marker = marker;

    // refine orientation based on plane ransac
    const Eigen::Quaterniond refined_orientation = extractOrientation(coefficients);
    refined_marker.rotation_x = refined_orientation.x();
    refined_marker.rotation_y = refined_orientation.y();
    refined_marker.rotation_z = refined_orientation.z();
    refined_marker.rotation_w = refined_orientation.w();

    // always refine position, because depth of board is also bad on larger distances
    if (camera_position.has_value())
    {
      const Eigen::Matrix<double, 3, 1> refined_translation =
          calculateRayPlaneIntersection(coefficients, camera_position.value(), marker);
      refined_marker.x = refined_translation.x();
      refined_marker.y = refined_translation.y();
      refined_marker.z = refined_translation.z();
    }

    refined_marker.refined_by_lidar = true;

    refined_markers.push_back(std::move(refined_marker));
  }

  return refined_markers;
}

pcl::PointCloud<AduulmPoint> RefinementToolbox::getDebugPCL() const
{
  return debug_pcl_;
}

std::tuple<double, double, double> RefinementToolbox::getPosStdDevAndTiltFromCov(const Detection& marker) const
{
  const Eigen::EigenSolver<Eigen::Matrix2d> solver(marker.pos_cov);
  const Eigen::VectorXcd w = solver.eigenvalues();
  const Eigen::MatrixXcd v = solver.eigenvectors();

  double w0 = w[0].real();
  double w1 = w[1].real();
  Eigen::Vector2d v0 = v.col(0).real();
  Eigen::Vector2d v1 = v.col(1).real();

  // swap std_dev
  if (w0 < w1)
  {
    std::swap(w0, w1);
    std::swap(v0, v1);
  }

  const double stddev_0 = sqrt(w0);  // stddev
  const double stddev_1 = sqrt(w1);

  const double tilt = std::atan2(v0[1], v0[0]);  // direction of the main axis
  return { stddev_0, stddev_1, tilt };
}

RefinementToolbox::PointCloudPtr RefinementToolbox::extractRelevantPoints(const PointCloudPtr& points,
                                                                          const Detection& marker) const
{
  PointCloudPtr filtered_points = std::make_shared<pcl::PointCloud<AduulmPoint>>();
  pcl::CropBox<AduulmPoint> crop_box_filter(true);
  crop_box_filter.setInputCloud(points);

  const auto [stddev_0, stddev_1, yaw] = getPosStdDevAndTiltFromCov(marker);

  // set long and lat by covariance
  const double sin_angle = sin(abs(marker.marker_angle_to_camera_ray));
  const double cos_angle = cos(abs(marker.marker_angle_to_camera_ray));
  const double long_search_dist = stddev_0 + params_->filter_distance.search_dist_next_marker * cos_angle +
                                  params_->filter_distance.search_dist_depth_marker * sin_angle;
  const double lat_search_dist = stddev_1 + params_->filter_distance.search_dist_next_marker * cos_angle -
                                 params_->filter_distance.search_dist_depth_marker * sin_angle;
  const Eigen::Vector4f lower_bound(
      -long_search_dist, -lat_search_dist, -params_->filter_distance.lower_filtering_distance_z, 1.0F);
  const Eigen::Vector4f upper_bound(
      long_search_dist, lat_search_dist, params_->filter_distance.upper_filtering_distance_z, 1.0F);

  crop_box_filter.setMin(lower_bound);
  crop_box_filter.setMax(upper_bound);

  const Eigen::Vector3f translation(marker.x, marker.y, marker.z);
  crop_box_filter.setTranslation(translation);
  const Eigen::Vector3f rpy(0.0, 0.0, yaw);
  crop_box_filter.setRotation(rpy);

  crop_box_filter.filter(*filtered_points);

  return filtered_points;
}

Eigen::Quaterniond extractOrientation(const pcl::ModelCoefficients& coefficients)
{
  // expected plane representation: Ax + By + Cz + D = 0
  Eigen::Vector3d plane_normal(coefficients.values[0], coefficients.values[1], coefficients.values[2]);
  Eigen::Vector3d plane_point(0, 0, -(coefficients.values[3] / coefficients.values[2]));

  // the normal vector of the plane should always point towards the lidar sensor
  const auto scalar_product = plane_normal.dot(plane_point);
  if (scalar_product > 0)
  {
    plane_normal *= -1;
  }

  const Eigen::Vector3d up_vec(0, 0, 1);
  const Eigen::Vector3d side_vec = plane_normal.cross(up_vec);
  const Eigen::Vector3d up_refined = plane_normal.cross(side_vec);

  Eigen::Matrix3d rotation_matrix;
  rotation_matrix.col(0) = plane_normal;
  rotation_matrix.col(1) = side_vec;
  rotation_matrix.col(2) = up_refined;

  return Eigen::Quaterniond(rotation_matrix);
}

Eigen::Vector3d calculateRayPlaneIntersection(const pcl::ModelCoefficients& coefficients,
                                              const Pos3D& camera_position,
                                              const lib::Detection& marker)
{
  // expected plane representation: Ax + By + Cz + D = 0
  const Eigen::Vector3d origin(camera_position.x, camera_position.y, camera_position.z);
  const Eigen::Vector3d target(marker.x, marker.y, marker.z);

  const auto ray = Eigen::ParametrizedLine<double, 3>::Through(origin, target);

  const Eigen::Vector3d plane_normal(coefficients.values[0], coefficients.values[1], coefficients.values[2]);
  const Eigen::Vector3d plane_point(0, 0, -(coefficients.values[3] / coefficients.values[2]));

  const Eigen::Hyperplane<double, 3> hyperplane(plane_normal.normalized(), plane_point);

  return ray.intersectionPoint(hyperplane);
  ;
}

}  // namespace capsule_pose_estimation::lib::lidar
