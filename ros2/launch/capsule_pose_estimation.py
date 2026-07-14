from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Optional
from aduulm_launch_lib_py import LaunchConfig, LogLevel
from aduulm_launch_py import execute_with_params
from aduulm_launch_lib_py.utils import asdict_filtered
from aduulm_tools_python.launch_utils import get_package_share_directory
from pathlib import Path


@dataclass(slots=True)
class BoardParameters:
    use_for_refinement: Optional[bool] = None
    checker_size: Optional[float] = None
    marker_size: Optional[float] = None
    num_cols: Optional[int] = None
    num_rows: Optional[int] = None
    main_marker_size: Optional[float] = None


@dataclass(slots=True)
class DebugParameters:
    show_markers: Optional[bool] = None
    show_rejected_markers: Optional[bool] = None
    show_poses: Optional[bool] = True
    broadcast_detections_as_transforms: Optional[bool] = None


@dataclass(slots=True)
class PositionMeasurementUncertaintyParams:
    base: Optional[float] = None
    distance_scaling: Optional[float] = None
    angle_scaling: Optional[float] = None

class OrientationMeasurementUncertaintyParams:
    est_angular_calib_error: Optional[float] = None
    est_xy_calib_error: Optional[float] = None
    est_pixel_error: Optional[int] = None


@dataclass(slots=True)
class CombinedMeasurementUncertaintyParameters:
    reduction_via_board: Optional[float] = None
    position: PositionMeasurementUncertaintyParams = field(default_factory=PositionMeasurementUncertaintyParams)
    orientation: OrientationMeasurementUncertaintyParams = field(default_factory=OrientationMeasurementUncertaintyParams)


@dataclass(slots=True)
class CPECameraParameters:
    launch: bool = True

    board: BoardParameters = field(default_factory=BoardParameters)
    debug: DebugParameters = field(default_factory=DebugParameters)
    meas_uncertainty: CombinedMeasurementUncertaintyParameters = field(
        default_factory=CombinedMeasurementUncertaintyParameters)

    use_gpu: Optional[bool] = None
    gpu_shared_mem_service_ns: str = "GpuSharedMemService"

    output_frame_id: str = "odom"

    aruco_dict: str = "4x4_50"
    single_marker_id_threshold: Optional[int] = None
    marker_size: Optional[float] = None

    min_marker_perimeter_rate: Optional[float] = None

    log_level: Optional[LogLevel] = LogLevel.Warning


@dataclass(slots=True)
class FilterDistance:
    search_dist_next_marker: Optional[float] = None
    search_dist_depth_marker: Optional[float] = None
    upper_z: Optional[float] = None
    lower_z: Optional[float] = None

@dataclass(slots=True)
class RansacParams:
    ransac_threshold: Optional[float] = None
    optimize_coefficients: Optional[bool] = None
    method_type: Optional[int] = None


@dataclass(slots=True)
class CPELidarParameters:
    launch: bool = True

    log_level: Optional[LogLevel] = LogLevel.Warning

    # distance parameters for point extraction
    filter_distance: FilterDistance = field(default_factory=FilterDistance)
    ransac_params: RansacParams = field(default_factory=RansacParams)

    # minimum distance (marker <-> lidar) that has to be exceeded for lidar refinement (due to minimum sensing distance
    # of lidars)
    min_distance: Optional[float] = None
    uncertainty_reduction: Optional[float] = None
    refine_boards: Optional[bool] = None
    prepare_debug_pcl: Optional[bool] = None


@dataclass(slots=True)
class CPETargetParameters:
    launch: bool = True
    target_config_file: Optional[str] = None

    log_level: Optional[LogLevel] = LogLevel.Warning


@dataclass(slots=True)
class CapsulePoseEstimationParameters:
    camera: CPECameraParameters = field(default_factory=CPECameraParameters)
    lidar: CPELidarParameters = field(default_factory=CPELidarParameters)
    target: CPETargetParameters = field(default_factory=CPETargetParameters)

    output_frame_id: str = "odom"

    # remappings
    topic_cpu_image: str = "image_rect"
    topic_gpu_image: str = "gpu_image_rect"
    topic_camera_info: str = "camera_info"
    marker_topic_detections: str = "marker_detections"
    topic_points: str = "points"
    refined_marker_topic_detections: str = "refined_marker_detections"
    topic_debug_pointcloud: str = "debug_pointcloud"
    target_topic_detections: str = "target_detections"


def gen_config(config: LaunchConfig, params: CapsulePoseEstimationParameters):
    config.insert_overrides(params)

    cam_params = asdict_filtered(params.camera)
    cam_params['output_frame_id'] = params.output_frame_id

    capsule_input_topic = params.marker_topic_detections

    if params.camera.launch:
        # camera node for marker detections
        node = config.add_node(
            name='capsule_marker_detector',
            package='capsule_pose_estimation',
            executable='camera_pose_estimator',
            handle_lifecycle=False,
            parameters=cam_params,
            # xterm=True,
            # gdb=True
        )

        marker_detection_pub = config.add_publisher(node, name="marker_detections", topic=params.marker_topic_detections)
        debug_pub = config.add_publisher(node, name="debug_image", topic="debug_image")

        if not params.camera.use_gpu:
            config.add_subscriber(node, name='image_rect', topic=params.topic_cpu_image, outputs=[marker_detection_pub,
                                                                                                  debug_pub])
            config.add_subscriber(node, name='camera_info', topic=params.topic_camera_info)
        else:
            config.add_subscriber(node, name='gpu_image_rect', topic=params.topic_gpu_image, outputs=[marker_detection_pub,
                                                                                                  debug_pub])
    # lidar node for refinement of marker detections
    if params.lidar.launch:
        capsule_input_topic = params.refined_marker_topic_detections

        lidar_node = config.add_node(
            name='capsule_marker_refiner',
            package='capsule_pose_estimation',
            executable='lidar_pose_refinement',
            handle_lifecycle=False,
            parameters=asdict_filtered(params.lidar),
        )

        refined_marker_pub = config.add_publisher(lidar_node, name="refined_marker_detections",
                                                  topic=params.refined_marker_topic_detections)
        debug_pcl_pub = config.add_publisher(lidar_node, name="debug_pointcloud",
                                                  topic=params.topic_debug_pointcloud)

        # currently doesn't support orchestrated replay
        config.add_subscriber(lidar_node, name='marker_detections', topic=params.marker_topic_detections,
                              outputs=[refined_marker_pub, debug_pcl_pub])
        config.add_subscriber(lidar_node, name='points', topic=params.topic_points, outputs=[])

    if params.target.launch:
        target_node = config.add_node(
            name='target_detection',
            package='capsule_pose_estimation',
            executable='target_detection',
            handle_lifecycle=False,
            parameters=asdict_filtered(params.target),
        )

        # currently doesn't support orchestrated replay
        target_detection_pub = config.add_publisher(target_node, name="target_detections",
                                                    topic=params.target_topic_detections)
        config.add_subscriber(target_node,
                              name="refined_marker_detections",
                              topic=capsule_input_topic,
                              outputs=[target_detection_pub])


if __name__ == "__main__":
    execute_with_params(gen_config)
