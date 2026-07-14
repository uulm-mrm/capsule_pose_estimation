/**
 * @file capsule_pose_estimation.hpp
 *
 * @brief Toolbox for the pose estimation of U-Shift II capsules.
 *
 * @author Oliver Schumann
 * Contact: oliver.schumann@uni-ulm.de
 */

#ifndef CAPSULE_POSE_ESTIMATION_LIB_HPP
#define CAPSULE_POSE_ESTIMATION_LIB_HPP

#include <iostream>
#include <optional>
#include <vector>

#include <opencv2/aruco.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>

#include "detection.hpp"

namespace capsule_pose_estimation::lib
{
struct BoardRefinementParams
{
  bool use_for_refinement = true;

  int num_columns = 9;
  int num_rows = 5;

  float checker_size = 0.12;
  float marker_size = 0.09;

  // Determines the maximum allowed distance between two any markers for being considered as part of the same charuco
  // board (thereby scaled with the board marker size in pixels)
  float gating_factor = 5.0;
};

struct DebuggingParameters
{
  bool show_markers = false;
  bool show_rejected_markers = false;
  bool show_marker_gating = false;
  bool show_poses = true;

  [[nodiscard]] bool any_set() const
  {
    return show_markers || show_rejected_markers || show_poses || show_marker_gating;
  }
};

struct Parameters
{
  cv::aruco::PredefinedDictionaryType dict_type = cv::aruco::PredefinedDictionaryType::DICT_4X4_50;

  /// Size of single aruco markers (size of markers forming a charuco board is specified separately).
  float marker_size = 0.18;
  /// All markers with at least this ID are expected to be single markers, while all IDs below this threshold
  /// are expected to be part of a charuco board.
  unsigned int single_marker_id_threshold = 47;

  // detection parameters
  double min_marker_perimeter_rate = 0.01;

  BoardRefinementParams board;
};

class ArucoToolbox
{
public:
  /**
   * Constructor.
   * @param params              Parameters which can only be changed at startup (re-instantiate toolbox if needed).
   * @param debuggingParameters Debug parameters which can be changed during runtime.
   */
  explicit ArucoToolbox(const Parameters& params, const std::shared_ptr<DebuggingParameters>& debug_params);

  /**
   *
   * @param image             Input image, only used for reading.
   * @param camera_matrix     Camera matrix required for projection.
   * @param distortion_coeffs Distortion coefficients for projection.
   * @return List of detected markers.
   *
   * @note Input image is non-const since const has no effect on cv::InputArray (by design of OpenCV).
   */
  std::vector<Detection> estimateMarkerPoses(cv::InputArray& image,
                                             const cv::Mat& camera_matrix,
                                             const cv::Mat& distortion_coeffs);

  [[nodiscard]] cv::Mat getDebugImage() const;

private:
  [[nodiscard]] Detection estimateMarkerPose(int id,
                                             const std::vector<cv::Point2f>& corners,
                                             const cv::Mat& camera_matrix,
                                             const cv::Mat& distortion_coeffs) const;

  [[nodiscard]] auto gate_markers(const std::vector<cv::Point2f>& main_marker_corners,
                                  const std::vector<int>& marker_ids,
                                  const std::vector<std::vector<cv::Point2f>>& marker_corners) const
      -> std::pair<std::vector<int>, std::vector<std::vector<cv::Point2f>>>;

  std::vector<Detection> estimateSingleMarkerPoses(cv::InputArray& image,
                                                   const std::vector<int>& marker_ids,
                                                   const std::vector<std::vector<cv::Point2f>>& marker_corners,
                                                   const cv::Mat& camera_matrix,
                                                   const cv::Mat& distortion_coeffs);
  std::vector<Detection> refineSingleMarkerPoses(cv::InputArray& image,
                                                 const std::vector<int>& marker_ids,
                                                 const std::vector<std::vector<cv::Point2f>>& marker_corners,
                                                 const cv::Mat& camera_matrix,
                                                 const cv::Mat& distortion_coeffs);

  [[nodiscard]] bool transformToMarkerCenter(cv::Vec3d& rotation_camera_coords,
                                             cv::Vec3d& translation_camera_coords,
                                             const cv::Mat& camera_matrix,
                                             const cv::Mat& distortion_coeffs,
                                             const std::vector<cv::Point2f>& marker_corners) const;
  /**
   * Transforms the coordinates from camera to generic sensor / vehicle coordinates.
   * @param translation_vec Translation vector of detected pose.
   * @param rotation_vec    Rotation vector of detected pose.
   * @return Board detection.
   */
  [[nodiscard]] static Detection createFromCameraCoordinates(const cv::Vec3d& translation_vec,
                                                             const cv::Vec3d& rotation_vec,
                                                             unsigned int id = 0);

  Parameters params_;
  std::shared_ptr<DebuggingParameters> debug_params_;

  // True in case the main marker covers an even number of board rows / cols, false otherwise. Determined based on the
  // provided sizes for the main marker / board marker.
  bool even_marker_ratio_;

  std::unique_ptr<cv::aruco::ArucoDetector> marker_detector_;
  std::unique_ptr<cv::aruco::CharucoDetector> board_detector_;

  std::shared_ptr<cv::aruco::Dictionary> dictionary_;

  std::shared_ptr<cv::aruco::CharucoBoard> board_;

  cv::Mat debug_image_;
};

}  // namespace capsule_pose_estimation::lib

#endif  // CAPSULE_POSE_ESTIMATION_LIB_HPP
