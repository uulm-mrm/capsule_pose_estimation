cmake_minimum_required(VERSION 3.10)
project(capsule_pose_estimation_lib)

# get aduulm cmake macros for coloring, package_version extraction and find_aduulm_package()
find_package(aduulm_cmake_tools REQUIRED)
get_version_string_from_package_xml(.)
show_build_info("library")
setup_git_hooks()

find_aduulm_package(aduulm_logger_lib REQUIRED)
find_aduulm_package(aduulm_measurement_types_lib REQUIRED)
find_package(OpenCV REQUIRED)

##########
# camera #
##########
set(CAM_LIBARY_NAME camera_detection_lib)
add_library(${CAM_LIBARY_NAME} SHARED
	src/logger_setup.cpp
	src/aruco_toolbox.cpp
)
add_library(${PROJECT_NAME}::${CAM_LIBARY_NAME} ALIAS ${CAM_LIBARY_NAME})

target_compile_features(${CAM_LIBARY_NAME} PRIVATE cxx_std_20)
target_compile_definitions(${CAM_LIBARY_NAME} PUBLIC "-D${package_name}_VERSION=\"${package_version}\"")

target_include_directories(${CAM_LIBARY_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)

target_include_directories(${CAM_LIBARY_NAME} SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${CAM_LIBARY_NAME} PRIVATE ${OpenCV_LIBS})

link_aduulm_package_targets(TARGET ${CAM_LIBARY_NAME}
  ACCESS PUBLIC
  PACKAGE_TARGETS
		aduulm_logger_lib::aduulm_logger_lib
)

#########
# lidar #
#########
find_package(PCL REQUIRED)

set(LIDAR_LIBRARY_NAME lidar_refinement_lib)
add_library(${LIDAR_LIBRARY_NAME} SHARED
		src/logger_setup.cpp
		src/lidar_refinement_toolbox.cpp
)
add_library(${PROJECT_NAME}::${LIDAR_LIBRARY_NAME} ALIAS ${LIDAR_LIBRARY_NAME})

target_compile_features(${LIDAR_LIBRARY_NAME} PRIVATE cxx_std_20)
target_compile_definitions(${LIDAR_LIBRARY_NAME} PUBLIC "-D${package_name}_VERSION=\"${package_version}\"")

target_include_directories(${LIDAR_LIBRARY_NAME} SYSTEM PUBLIC ${PCL_INCLUDE_DIRS})

target_include_directories(${LIDAR_LIBRARY_NAME}
		PUBLIC
		$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)

target_link_libraries(${LIDAR_LIBRARY_NAME} PRIVATE ${PCL_LIBRARIES}) # TODO: PRIVATE should be sufficient?
link_aduulm_package_targets(TARGET ${LIDAR_LIBRARY_NAME}
		ACCESS PUBLIC
		PACKAGE_TARGETS
		aduulm_logger_lib::aduulm_logger_lib
		aduulm_measurement_types_lib::aduulm_measurement_types_lib
)

# combined library (only needed because aduulm_cmake_tools will try to autolink this library based on the node to lib
# dependency expressed in the package.xml)
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} INTERFACE ${CAM_LIBARY_NAME} ${LIDAR_LIBRARY_NAME})

set(PACKAGE_LIBRARY_VERSION ${package_version})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
		"${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
		VERSION "${PACKAGE_LIBRARY_VERSION}"
		COMPATIBILITY AnyNewerVersion
)

install(TARGETS ${CAM_LIBARY_NAME} ${PROJECT_NAME} ${LIDAR_LIBRARY_NAME}
		EXPORT ${PROJECT_NAME}Targets
		INCLUDES DESTINATION ${INCLUDE_INSTALL_DIR}
		LIBRARY DESTINATION ${LIB_INSTALL_DIR} COMPONENT Runtime
		ARCHIVE DESTINATION ${LIB_INSTALL_DIR} COMPONENT Development
		RUNTIME DESTINATION ${BIN_INSTALL_DIR} COMPONENT Runtime
		PUBLIC_HEADER DESTINATION ${INCLUDE_INSTALL_DIR} COMPONENT Development
		BUNDLE DESTINATION ${BIN_INSTALL_DIR} COMPONENT Runtime
)

configure_package_config_file(
		"${PROJECT_SOURCE_DIR}/config.cmake.in" # this file defines the include dirs and libs and may need to be adapted for your package/library
		"${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
		INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/${PROJECT_NAME}
		PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR BIN_INSTALL_DIR
)

## Create targets file with all targets (enables to link against single targets)
install(EXPORT ${PROJECT_NAME}Targets
		DESTINATION ${LIB_INSTALL_DIR}/cmake/${PROJECT_NAME}
		NAMESPACE ${PROJECT_NAME}::
)

# install CMake config and version file
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
		"${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
		DESTINATION ${LIB_INSTALL_DIR}/cmake/${PROJECT_NAME})

## install header files
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${INCLUDE_INSTALL_DIR})
