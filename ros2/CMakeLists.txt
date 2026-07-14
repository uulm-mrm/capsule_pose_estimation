cmake_minimum_required(VERSION 3.10)
project(capsule_pose_estimation)

find_package(aduulm_cmake_tools)

aduulm_init_ros2_package()

# node for camera detections
set(CAMERA_NODE camera_pose_estimator)
aduulm_add_ros2_executable(${CAMERA_NODE}
    src/camera_detection.cpp
)

target_link_libraries(${CAMERA_NODE} PUBLIC
    ${OpenCV_LIBS}
)

link_aduulm_package_targets(TARGET ${CAMERA_NODE}
    ACCESS PUBLIC
    PACKAGE_TARGETS
    aduulm_logger_lib::aduulm_logger_lib
    capsule_pose_estimation_lib::camera_detection_lib
)
target_compile_features(${CAMERA_NODE} PRIVATE cxx_std_20)

# node for lidar refinement of camera detections
set(LIDAR_NODE lidar_pose_refinement)
aduulm_add_ros2_executable(${LIDAR_NODE}
        src/lidar_refinement.cpp
)

link_aduulm_package_targets(TARGET ${LIDAR_NODE}
        ACCESS PUBLIC
        PACKAGE_TARGETS
        aduulm_logger_lib::aduulm_logger_lib
        capsule_pose_estimation_lib::lidar_refinement_lib
)
target_compile_features(${LIDAR_NODE} PRIVATE cxx_std_20)

# node for target detection with refined markers
set(TARGET_NODE target_detection)
aduulm_add_ros2_executable(${TARGET_NODE}
        src/target_detection.cpp
)

target_link_libraries(${TARGET_NODE} PUBLIC
        yaml-cpp
)

link_aduulm_package_targets(TARGET ${TARGET_NODE}
        ACCESS PUBLIC
        PACKAGE_TARGETS
        aduulm_logger_lib::aduulm_logger_lib
)
target_compile_features(${TARGET_NODE} PRIVATE cxx_std_20)


# install launch files for aduulm_launch
aduulm_install_ros2_python_code("${PROJECT_NAME}_launch" PACKAGE_DIR ${CMAKE_CURRENT_LIST_DIR}/launch)

aduulm_create_ros2_package()
setup_git_hooks()


