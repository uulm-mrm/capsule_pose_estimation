#ifndef CAPSULE_POSE_ESTIMATION_LIDAR_REFINEMENT_TOOLBOX_HPP
#define CAPSULE_POSE_ESTIMATION_LIDAR_REFINEMENT_TOOLBOX_HPP

#include <memory>
#include <vector>

#include <aduulm_measurement_types_lib/point_cloud_types.hpp>
#include <pcl/point_cloud.h>
#include <pcl/sample_consensus/method_types.h>

#include "detection.hpp"

namespace capsule_pose_estimation::lib::lidar
{
using AduulmPoint = aduulm_measurement_types_lib::AduulmPoint;

struct FilterDistance
{
  double search_dist_next_marker = 0.5;
  double search_dist_depth_marker = 0.05;
  double lower_filtering_distance_z = 0.35F;  // unit: meters
  double upper_filtering_distance_z = 0.35F;  // unit: meters
};

struct RansacParams
{
  double ransac_threshold = 0.1;
  bool optimize_coefficients = true;
  //  SAC_RANSAC  = 0;
  //  SAC_LMEDS   = 1;
  //  SAC_MSAC    = 2;
  //  SAC_RRANSAC = 3;
  //  SAC_RMSAC   = 4;
  //  SAC_MLESAC  = 5;
  //  SAC_PROSAC  = 6;
  int method_type = pcl::SAC_RANSAC;
};

struct RefinementParams
{
  // given in marker coordinates (x towards cam, z upwards), radius measured from marker center

  FilterDistance filter_distance;
  RansacParams ransac_params;

  double min_distance = 0.0;
  double uncertainty_reduction = 0.15;
  bool refine_boards = true;
  bool prepare_debug_pcl = true;
};

struct Pos3D
{
  float x;
  float y;
  float z;
};

class RefinementToolbox
{
public:
  using PointCloudPtr = std::shared_ptr<pcl::PointCloud<AduulmPoint>>;
  using PointCloudConstPtr = std::shared_ptr<pcl::PointCloud<const AduulmPoint>>;

  explicit RefinementToolbox(std::shared_ptr<RefinementParams> params);

  [[nodiscard]] pcl::PointCloud<AduulmPoint> getDebugPCL() const;

  // Note: all arguments need to be provided with respect to the same reference frame
  [[nodiscard]] std::vector<Detection> refineMarkers(const std::vector<Detection>& markers,
                                                     const PointCloudPtr& points,
                                                     const std::string& camera_frame,
                                                     std::optional<Pos3D> camera_position = std::nullopt);

private:
  std::tuple<double, double, double> getPosStdDevAndTiltFromCov(const Detection& marker) const;

  PointCloudPtr extractRelevantPoints(const PointCloudPtr& points, const Detection& marker) const;

  std::shared_ptr<RefinementParams> params_;
  pcl::PointCloud<AduulmPoint> debug_pcl_;
};
}  // namespace capsule_pose_estimation::lib::lidar

#endif  // CAPSULE_POSE_ESTIMATION_LIDAR_REFINEMENT_TOOLBOX_HPP
