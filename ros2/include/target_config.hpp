/**
 * @file capsule_config.hpp
 *
 * @brief configs classes of the target detection
 *
 * @author Oliver Schumann
 * Contact: oliver.schumann@uni-ulm.de
 *
 */
#ifndef TARGET_CONFIG_HPP
#define TARGET_CONFIG_HPP

#include <vector>

#include <Eigen/Dense>

#include <yaml-cpp/yaml.h>

namespace capsule_pose_estimation
{

class MarkerConfig
{
public:
  explicit MarkerConfig(const std::string& key, const YAML::Node& value)
  {
    // Get marker info
    id_ = stoi(key);
    side_ = value["side"].as<std::string>();

    // Get trafos
    // capsule -> marker
    auto alignment_vector = value["alignment"].as<std::vector<double>>();
    target2marker_ = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(alignment_vector.data());
    // marker -> capsule
    marker2target_ = target2marker_.inverse();
  }

  int id_;
  std::string side_;
  Eigen::Matrix4d target2marker_;
  Eigen::Matrix4d marker2target_;
};

class TargetConfig
{
public:
  /**
   * Create TargetConfig that contains several markers
   * @param key
   * @param target_yaml
   */
  explicit TargetConfig(const std::string& key, const YAML::Node& target_yaml)
  {
    // Get target info
    id_ = stoi(key);
    type_ = target_yaml["type"].as<std::string>();
    name_ = target_yaml["name"].as<std::string>();

    // Get all existing markers
    for (const auto& key_value : target_yaml["markers"])
    {
      const auto marker_key = key_value.first.as<std::string>();
      const YAML::Node& marker_yaml = key_value.second;

      MarkerConfig marker_config(marker_key, marker_yaml);

      marker_configs_.try_emplace(marker_config.id_, std::move(marker_config));
    }
  }

  int id_;
  std::string type_;
  std::string name_;
  std::map<int, MarkerConfig> marker_configs_;
};

class Targets
{
public:
  /**
   * Read all targets from file
   * @param config_path
   */
  explicit Targets(const std::string& config_path)
  {
    try
    {
      const YAML::Node config_yaml = YAML::LoadFile(config_path);

      // Get all existing targets
      for (const auto& key_value : config_yaml["targets"])
      {
        const auto target_key = key_value.first.as<std::string>();
        const YAML::Node& target_yaml = key_value.second;

        target_configs_.emplace_back(target_key, target_yaml);
      }
    }
    catch (YAML::BadFile& e)
    {
      std::string error = "Could not read targets config from " + config_path;
      LOG_ERR(error);
      throw std::runtime_error(error);
    }
  }

  std::vector<TargetConfig> target_configs_;
};
}  // namespace capsule_pose_estimation
#endif  // TARGET_CONFIG_HPP
