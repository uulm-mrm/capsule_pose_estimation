/**
 * @file capsule_pose_estimation.cpp
 *
 * @brief pose estimation of charuco boards
 *
 * @ingroup image_processing_lib
 *
 * @author Oliver Schumann
 * Contact: oliver.schumann@uni-ulm.de
 *
 */

#include <opencv2/calib3d.hpp>
#include "aruco_toolbox.hpp"
#include "logger_setup.hpp"

static constexpr float FRAME_AXES_THICKNESS = 0.1F;
static constexpr float MAX_DISTANCE_TO_ESTIMATED_MARKER_CENTER_PX = 5.0F;  // unit: pixel

static const std::vector<cv::Scalar> GATING_COLORS = { cv::Scalar(255, 0, 255),
                                                       cv::Scalar(125, 125, 255),
                                                       cv::Scalar(0, 125, 125) };
static const cv::Scalar IGNORED_MARKER_COLOR(0, 255, 0);

namespace capsule_pose_estimation::lib
{

ArucoToolbox::ArucoToolbox(const Parameters& params, const std::shared_ptr<DebuggingParameters>& debug_params)
  : params_(params), debug_params_(debug_params)

{
  dictionary_ = std::make_shared<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(params_.dict_type));

  cv::aruco::DetectorParameters detector_params;
  detector_params.minMarkerPerimeterRate = params_.min_marker_perimeter_rate;
  detector_params.cornerRefinementMethod = cv::aruco::CornerRefineMethod::CORNER_REFINE_SUBPIX;

  marker_detector_ = std::make_unique<cv::aruco::ArucoDetector>(*dictionary_, detector_params);

  if (params_.board.use_for_refinement)
  {
    // sanity checks
    assert(params_.board.num_columns > 0 && params_.board.num_rows > 0 && "invalid charuco board configuration");

    const auto& board_params = params_.board;
    board_ = std::make_shared<cv::aruco::CharucoBoard>(cv::Size(board_params.num_columns, board_params.num_rows),
                                                       board_params.checker_size,
                                                       board_params.marker_size,
                                                       *dictionary_);

    // camera parameters are default initialized because we override them with every new image
    board_detector_ =
        std::make_unique<cv::aruco::CharucoDetector>(*board_, cv::aruco::CharucoParameters(), detector_params);

    const float marker_ratio = std::round(params_.marker_size / params_.board.marker_size);
    even_marker_ratio_ = static_cast<int>(marker_ratio) % 2 == 0;
  }
}

std::vector<Detection> ArucoToolbox::estimateMarkerPoses(cv::InputArray& image,
                                                         const cv::Mat& camera_matrix,
                                                         const cv::Mat& distortion_coeffs)
{
  // detect markers
  std::vector<int> marker_ids;
  std::vector<std::vector<cv::Point2f>> marker_corners;
  std::vector<std::vector<cv::Point2f>> rejected_markers;

  marker_detector_->detectMarkers(image, marker_corners, marker_ids, rejected_markers);

  // optional visualization to help with 'live' debugging
  if (debug_params_->show_rejected_markers && !rejected_markers.empty())
  {
    // image has to be BGR, otherwise axes of drawFrameAxes will have switched colors
    cv::cvtColor(image, debug_image_, cv::COLOR_RGB2BGR);
    cv::aruco::drawDetectedMarkers(debug_image_, rejected_markers, cv::Mat(), cv::Scalar(0, 0, 255));
  }

  if (debug_params_->show_markers && !marker_corners.empty())
  {
    // image has to be BGR, otherwise axes of drawFrameAxes will have switched colors
    cv::cvtColor(image, debug_image_, cv::COLOR_RGB2BGR);
    cv::aruco::drawDetectedMarkers(debug_image_, marker_corners, marker_ids, cv::Scalar(0, 0, 255));
  }

  if (params_.board.use_for_refinement)
  {
    return refineSingleMarkerPoses(image, marker_ids, marker_corners, camera_matrix, distortion_coeffs);
  }

  return estimateSingleMarkerPoses(image, marker_ids, marker_corners, camera_matrix, distortion_coeffs);
}

cv::Mat ArucoToolbox::getDebugImage() const
{
  return debug_image_;
}

std::vector<Detection>
ArucoToolbox::estimateSingleMarkerPoses(cv::InputArray& image,
                                        const std::vector<int>& marker_ids,
                                        const std::vector<std::vector<cv::Point2f>>& marker_corners,
                                        const cv::Mat& camera_matrix,
                                        const cv::Mat& distortion_coeffs)
{
  std::vector<Detection> detections;

  // preparation of debug image
  if (debug_params_->show_poses)
  {
    // image has to be BGR, otherwise axes of drawFrameAxes will have switched colors
    cv::cvtColor(image, debug_image_, cv::COLOR_RGB2BGR);
    image.copyTo(debug_image_);
  }

  for (auto i = 0U; i < marker_ids.size(); i++)
  {
    const auto id = marker_ids.at(i);
    const auto& corners = marker_corners.at(i);

    if (id < params_.single_marker_id_threshold)
    {
      // based on the ID we assume that this marker is part of a charuco board and thus ignore it here
      continue;
    }

    detections.emplace_back(estimateMarkerPose(id, corners, camera_matrix, distortion_coeffs));
  }

  return detections;
}

Detection ArucoToolbox::estimateMarkerPose(const int id,
                                           const std::vector<cv::Point2f>& corners,
                                           const cv::Mat& camera_matrix,
                                           const cv::Mat& distortion_coeffs) const
{
  // we assume the same size for all single markers
  const float half_marker_length = params_.marker_size / 2;

  cv::Mat object_points(4, 1, CV_32FC3);
  object_points.at<cv::Vec3f>(0, 0) = cv::Vec3f(-half_marker_length, half_marker_length, 0);
  object_points.at<cv::Vec3f>(0, 1) = cv::Vec3f(half_marker_length, half_marker_length, 0);
  object_points.at<cv::Vec3f>(0, 2) = cv::Vec3f(half_marker_length, -half_marker_length, 0);
  object_points.at<cv::Vec3f>(0, 3) = cv::Vec3f(-half_marker_length, -half_marker_length, 0);

  cv::Vec3d rotation_camera_coords = cv::Vec3d::zeros();
  cv::Vec3d translation_camera_coords = cv::Vec3d::zeros();

  solvePnP(object_points, corners, camera_matrix, distortion_coeffs, rotation_camera_coords, translation_camera_coords);

  if (debug_params_->show_poses)
  {
    const std::vector<int> marker_id_vector = { id };
    const std::vector<std::vector<cv::Point2f>> corner_vec_of_vec = { corners };

    cv::aruco::drawDetectedMarkers(debug_image_, corner_vec_of_vec, marker_id_vector, cv::Scalar(0, 0, 255));
    cv::drawFrameAxes(debug_image_,
                      camera_matrix,
                      distortion_coeffs,
                      rotation_camera_coords,
                      translation_camera_coords,
                      FRAME_AXES_THICKNESS);
  }

  return createFromCameraCoordinates(translation_camera_coords, rotation_camera_coords, id);
}

auto ArucoToolbox::gate_markers(const std::vector<cv::Point2f>& main_marker_corners,
                                const std::vector<int>& marker_ids,
                                const std::vector<std::vector<cv::Point2f>>& marker_corners) const
    -> std::pair<std::vector<int>, std::vector<std::vector<cv::Point2f>>>
{
  // reduce marker corners to center point for easier comparison
  std::vector<cv::Point2f> marker_center_points(marker_ids.size());

  for (auto i = 0; i < marker_corners.size(); i++)
  {
    cv::Point2f center(0, 0);

    for (const auto& p : marker_corners.at(i))
    {
      center.x += p.x;
      center.y += p.y;
    }

    center.x /= 4;
    center.y /= 4;

    marker_center_points.at(i) = center;
  }

  // Note: the four corner points may be arbitrarily ordered. Hence, we rely on the fact that the marker is a square
  //       for determining the main marker edge length in px
  cv::Point2f main_marker_center(0, 0);
  for (const auto& corner : main_marker_corners)
  {
    main_marker_center.x += corner.x;
    main_marker_center.y += corner.y;
  }

  main_marker_center.x /= 4;
  main_marker_center.y /= 4;

  // heuristic for max distance between center points
  const float main_marker_size_px =
      static_cast<float>(cv::norm(main_marker_center - main_marker_corners.at(0)) / std::cos(M_PI / 4));
  const float board_marker_size_px = main_marker_size_px * params_.board.marker_size / params_.marker_size;

  std::vector<int> gated_marker_ids;
  std::vector<std::vector<cv::Point2f>> gated_marker_corners;

  // copy the ids and corners to be able to work on them
  std::vector<int> remaining_marker_ids = marker_ids;
  std::vector<std::vector<cv::Point2f>> remaining_marker_corners = marker_corners;

  // DBSCAN like iterative adding of markers based on the principal of 'core' points / markers
  std::vector<cv::Point2f> center_inliers = main_marker_corners;
  std::vector<float> inlier_edge_lengths_px(center_inliers.size(), board_marker_size_px);
  bool added_new_markers = true;
  while (added_new_markers)
  {
    added_new_markers = false;

    for (auto marker_idx = 0U; marker_idx < marker_center_points.size() && !marker_center_points.empty(); marker_idx++)
    {
      const auto& center = marker_center_points.at(marker_idx);

      for (auto i = 0U; i < center_inliers.size(); i++)
      {
        const auto& inlier = center_inliers.at(i);
        const auto& max_center_dist_px = inlier_edge_lengths_px.at(i) * params_.board.gating_factor;

        if (cv::norm(inlier - center) <= max_center_dist_px)
        {
          added_new_markers = true;

          // remembering the edge length of the small marker for the threshold decision helps in scenarios where
          // the board is viewed with a low angle from the side
          const float marker_size_px =
              static_cast<float>(cv::norm(center - remaining_marker_corners.at(marker_idx).at(0)) / std::cos(M_PI / 4));

          center_inliers.push_back(center);
          inlier_edge_lengths_px.push_back(marker_size_px);

          gated_marker_ids.push_back(remaining_marker_ids.at(marker_idx));
          gated_marker_corners.push_back(remaining_marker_corners.at(marker_idx));

          // avoid copying the whole vector while for removing a single item by swapping it with the last
          // element and reducing the size by one
          marker_center_points.at(marker_idx) = marker_center_points.back();
          marker_center_points.resize(marker_center_points.size() - 1);
          remaining_marker_ids.at(marker_idx) = remaining_marker_ids.back();
          remaining_marker_ids.resize(remaining_marker_ids.size() - 1);
          remaining_marker_corners.at(marker_idx) = remaining_marker_corners.back();
          remaining_marker_corners.resize(remaining_marker_corners.size() - 1);

          break;
        }
      }

      if (added_new_markers)
      {
        break;
      }
    }
  }

  return { gated_marker_ids, gated_marker_corners };
}

std::vector<Detection>
ArucoToolbox::refineSingleMarkerPoses(cv::InputArray& image,
                                      const std::vector<int>& marker_ids,
                                      const std::vector<std::vector<cv::Point2f>>& marker_corners,
                                      const cv::Mat& camera_matrix,
                                      const cv::Mat& distortion_coeffs)
{
  // preparation of debug image
  if (debug_params_->show_poses || debug_params_->show_marker_gating)
  {
    // image has to be BGR, otherwise axes of drawFrameAxes will have switched colors
    cv::cvtColor(image, debug_image_, cv::COLOR_RGB2BGR);
  }

  // distinguish between single markers and board markers
  std::vector<int> single_marker_ids;
  std::vector<std::vector<cv::Point2f>> single_marker_corners;

  std::vector<int> board_marker_ids;
  std::vector<std::vector<cv::Point2f>> board_marker_corners;

  for (auto idx = 0U; idx < marker_ids.size(); idx++)
  {
    if (marker_ids.at(idx) < params_.single_marker_id_threshold)
    {
      board_marker_ids.emplace_back(marker_ids.at(idx));
      board_marker_corners.emplace_back(marker_corners.at(idx));
    }
    else
    {
      single_marker_ids.emplace_back(marker_ids.at(idx));
      single_marker_corners.emplace_back(marker_corners.at(idx));
    }
  }

  // update camera intrinsics (could have changed, e.g. because resolution of camera has been increased)
  cv::aruco::CharucoParameters charuco_params;
  charuco_params.cameraMatrix = camera_matrix;
  charuco_params.distCoeffs = distortion_coeffs;

  board_detector_->setCharucoParameters(charuco_params);

  // search for a board around each single marker
  std::vector<Detection> detections;

  if (debug_params_->show_marker_gating)
  {
    // draw all markers first and add the gated ones with their respective color later on top of that
    cv::aruco::drawDetectedMarkers(debug_image_, board_marker_corners, {}, IGNORED_MARKER_COLOR);
  }

  for (auto i = 0U; i < single_marker_ids.size(); i++)
  {
    const int main_marker_id = single_marker_ids.at(i);
    const auto main_marker_corners = single_marker_corners.at(i);

    // filter markers based on distance to considered single marker
    const auto& [gated_marker_ids, gated_marker_corners] =
        gate_markers(main_marker_corners, board_marker_ids, board_marker_corners);

    if (!gated_marker_ids.empty())
    {
      if (debug_params_->show_marker_gating)
      {
        const auto& color = GATING_COLORS.at(i % GATING_COLORS.size());
        cv::aruco::drawDetectedMarkers(debug_image_, gated_marker_corners, {}, color);

        // draw the main marker with the same color to highlight the gating
        const std::vector<int> marker_id_vector = { main_marker_id };
        const std::vector<std::vector<cv::Point2f>> corner_vec_of_vec = { main_marker_corners };

        cv::aruco::drawDetectedMarkers(debug_image_, corner_vec_of_vec, marker_id_vector, color);
      }
      else if (debug_params_->show_poses)
      {
        cv::aruco::drawDetectedMarkers(
            debug_image_, gated_marker_corners, {}, GATING_COLORS.at(i % GATING_COLORS.size()));
      }
    }

    bool valid = false;
    std::vector<cv::Point2f> charuco_corners;
    std::vector<int> charuco_ids;
    cv::Vec3d rotation_camera_coords = cv::Vec3d::zeros();
    cv::Vec3d translation_camera_coords = cv::Vec3d::zeros();
    if (!gated_marker_ids.empty())
    {
      // detect charuco board and estimate pose
      board_detector_->detectBoard(image, charuco_corners, charuco_ids, gated_marker_corners, gated_marker_ids);

      // DLT algorithm requires at least 6 points for pose estimation from 3D-2D point correspondences
      if (charuco_corners.size() < 6)
      {
        valid = false;
      }
      else
      {
        valid = cv::aruco::estimatePoseCharucoBoard(charuco_corners,
                                                    charuco_ids,
                                                    board_,
                                                    camera_matrix,
                                                    distortion_coeffs,
                                                    rotation_camera_coords,
                                                    translation_camera_coords);
      }
    }

    // map the corner reference obtained from the board to the marker center
    // --> may fail in case the marker center obtained via the board is too far away from the center of the detected
    //     marker
    if (!valid ||
        !transformToMarkerCenter(
            rotation_camera_coords, translation_camera_coords, camera_matrix, distortion_coeffs, main_marker_corners))
    {
      // unable to refine pose using a charuco board, thus, falling back to normal pose estimation
      LOG_INF("Falling back to normal pose estimation for single marker with id " << main_marker_id);
      detections.emplace_back(
          estimateMarkerPose(single_marker_ids.at(i), single_marker_corners.at(i), camera_matrix, distortion_coeffs));
      continue;
    }

    if (debug_params_->show_poses)
    {
      cv::drawFrameAxes(debug_image_,
                        camera_matrix,
                        distortion_coeffs,
                        rotation_camera_coords,
                        translation_camera_coords,
                        FRAME_AXES_THICKNESS);
    }

    detections.emplace_back(
        createFromCameraCoordinates(translation_camera_coords, rotation_camera_coords, main_marker_id));
    detections.back().refined_by_board = true;
  }

  return detections;
}

Detection ArucoToolbox::createFromCameraCoordinates(const cv::Vec3d& translation_vec,
                                                    const cv::Vec3d& rotation_vec,
                                                    const unsigned int id)
{
  Detection detection;
  detection.x = translation_vec[0];
  detection.y = translation_vec[1];
  detection.z = translation_vec[2];

  const auto quat = cv::Quatd::createFromRvec(cv::Vec3d(rotation_vec[0], rotation_vec[1], rotation_vec[2]));
  detection.rotation_w = quat.w;
  detection.rotation_x = quat.x;
  detection.rotation_y = quat.y;
  detection.rotation_z = quat.z;

  detection.id = id;

  return detection;
}

bool ArucoToolbox::transformToMarkerCenter(cv::Vec3d& rotation_camera_coords,
                                           cv::Vec3d& translation_camera_coords,
                                           const cv::Mat& camera_matrix,
                                           const cv::Mat& distortion_coeffs,
                                           const std::vector<cv::Point2f>& marker_corners) const
{
  // Note: even though we assume the main marker is perfectly aligned to the surrounding charuco board there are two
  //       different cases which have to be distinguished:
  //       1) main marker covers an even number of cells within the board layout
  //          --> the refined marker position is the chessboard corner with the smallest distance to the marker center
  //       2) main marker covers an odd number of cells within the board layout
  //          --> consider offset of half cell size for comparison points

  // determine marker center
  cv::Point2f center = { 0, 0 };
  for (const auto& corner : marker_corners)
  {
    center.x += corner.x;
    center.y += corner.y;
  }
  center /= 4;

  // transform world points of the checker corners into their corresponding image points
  std::vector<cv::Point3f> corner_points;
  std::vector<cv::Point2f> image_points;

  const float offset = (even_marker_ratio_) ? 0.0F : 0.5F;

  for (auto row = 0; row < params_.board.num_rows; row++)
  {
    for (auto col = 0; col < params_.board.num_columns; col++)
    {
      // Note: since the points are projected into the camera coordinate system, the distance from the camera
      //       (z-coordinate within the camera coordinate system) doesn't matter
      corner_points.emplace_back(params_.board.checker_size * (static_cast<float>(col) + offset),
                                 params_.board.checker_size * (static_cast<float>(row) + offset),
                                 0);
    }
  }

  cv::projectPoints(
      corner_points, rotation_camera_coords, translation_camera_coords, camera_matrix, distortion_coeffs, image_points);

  unsigned int min_row = 0;
  unsigned int min_col = 0;
  double min_distance = std::numeric_limits<float>::max();
  for (auto row = 0; row < params_.board.num_rows; row++)
  {
    for (auto col = 0; col < params_.board.num_columns; col++)
    {
      const auto idx = row * params_.board.num_columns + col;

      const double distance = cv::norm(image_points.at(idx) - center);

      if (distance < min_distance)
      {
        min_distance = distance;
        min_row = row;
        min_col = col;
      }
    }
  }

  if (min_distance > MAX_DISTANCE_TO_ESTIMATED_MARKER_CENTER_PX)
  {
    // our estimated board pose is likely wrong --> ignore it
    LOG_INF("Aborting refinement using board pose due to large difference to detected marker center ("
            << min_distance << "px vs. " << MAX_DISTANCE_TO_ESTIMATED_MARKER_CENTER_PX << ")");
    return false;
  }

  // modify rotation of the coordinate system to match the original marker coordinate system
  // - expected board frame: x -> board right, y -> board down, z -> away from camera
  // - intended marker frame: x -> board right, y -> board top, z -> towards camera
  cv::Quatd orientation_offset =
      cv::Quatd::createFromEulerAngles(cv::Vec3d{ M_PI, 0, 0 }, cv::QuatEnum::EulerAnglesType::INT_XYZ);

  cv::Quatd new_orientation = cv::Quatd::createFromRvec(rotation_camera_coords) * orientation_offset;

  rotation_camera_coords = new_orientation.toRotVec();

  // modify translation to match the marker center by applying an offset within the board coordinate system
  const cv::Vec3d offset_board_coordinates = { (offset + static_cast<float>(min_col)) * params_.board.checker_size,
                                               -(offset + static_cast<float>(min_row)) * params_.board.checker_size,
                                               0 };

  const auto rotated_offset = new_orientation.toRotMat3x3() * offset_board_coordinates;

  translation_camera_coords += rotated_offset;

  return true;
}
}  // namespace capsule_pose_estimation::lib
