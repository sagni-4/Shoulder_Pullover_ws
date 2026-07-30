// Ego-relative shoulder centerline waypoint generation -- C++ port for speed.
//
// Paints the live LiDAR point cloud with shoulder_detection_node's 2D
// shoulder mask (PointPainting-style sequential fusion -- Vora, Lang, Helou,
// Beijbom, "PointPainting: Sequential Fusion for 3D Object Detection",
// IEEE/CVF CVPR 2020; the same idea Autoware itself ships as
// autoware_image_projection_based_fusion's pointpainting_fusion package),
// simplified here to a single-class geometric extraction rather than a
// learned 3D detector. All extrinsics come from live TF lookups (not
// hardcoded matrices), since the camera/LiDAR->base_link transforms are
// already static-broadcast by the carla_sensor_kit URDF.
//
// Pipeline per synchronized (image, mask, lidar) frame:
//   1. Get a lateral sample per forward-distance bin, from one of two
//      sources (bin_source param):
//      - "mask_ipm" (default): per sampled image row, find the largest
//        contiguous run of shoulder-classified columns, take its midpoint,
//        and back-project that pixel to base_link via ground-plane (z=0)
//        inverse perspective mapping. Dense/robust regardless of how narrow
//        the real shoulder is; assumes locally flat ground.
//      - "lidar": project LiDAR points into the mask's pixel frame, keep the
//        ones landing on the shoulder class, transform into base_link and
//        drop any outside a ground-height band. Sparse -- only a handful of
//        rays land on a narrow shoulder per frame, which was the actual
//        source of jitter/off-centering in earlier iterations, not the
//        per-bin aggregation method.
//   2. Bin by forward distance, take a lateral center per bin
//      (centerline_estimator: extent-midpoint of the samples, or median),
//      density-gated.
//   4. Fit a natural cubic spline (uniform-knot parametrization over the
//      valid bin sequence -- see cubic_spline.hpp) through the bin
//      centroids, resample at fixed arclength spacing for waypoints with
//      heading (first derivative) and signed curvature
//      kappa = (x'y'' - y'x'') / (x'^2+y'^2)^1.5 (standard Frenet-frame form).
//   5. Light exponential moving average per arclength step, frame to frame.
//
// Publishes:
//   - nav_msgs/Path (frame_id=output_frame/base_link) -- the local ego-
//     relative window, re-fit from scratch every frame (for a local
//     planner/controller; RViz's Path display only ever shows the latest
//     message, so watching this alone looks like the line "moves with the
//     vehicle" -- that's by design for this one).
//   - nav_msgs/Path (frame_id=map_frame/map) -- an accumulating trail: each
//     frame's local waypoints are transformed into map frame and appended
//     (distance-gated to avoid duplicate points from overlapping frames),
//     growing into a persistent world-anchored line the same way a planned
//     trajectory reference line does.
//   - a debug image with the painted LiDAR points + reprojected centerline,
//     for visual sanity-checking in RViz without a 3D view.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "shoulder_centerline/cubic_spline.hpp"

namespace
{

struct Waypoint
{
  double x, y, z, yaw, kappa;
};

// Every raw map-frame sample ever observed for one world-frame bin (see the
// accumulate_path_ comment in the constructor for why this is keyed by
// traveled arclength rather than a 2D grid cell).
struct WorldBinSamples
{
  std::vector<double> x, y, z;
};

double median(std::vector<double> v)
{
  const std::size_t n = v.size();
  const std::size_t mid = n / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  double m = v[mid];
  if (n % 2 == 0) {
    std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
    m = 0.5 * (m + v[mid - 1]);
  }
  return m;
}

// A per-frame (or per-world-bin) median of raw painted-point lateral offsets
// is a *density-weighted* centroid: whichever edge of the shoulder happens to
// get more LiDAR hits (scan-pattern/incidence-angle dependent, not uniform
// across the lane's width) pulls the estimate toward it, off the true
// geometric center. Classic lane-detection literature instead finds the
// left/right boundary of the classified region and takes their midpoint
// (e.g. the sliding-window/histogram lane-finding technique, and "the
// lane-center line calculated as the average between the left and right
// lane boundary" in lane-sensing patents/surveys) -- that's what this does,
// using a trimmed (not bare) min/max so a single outlier point doesn't drag
// one edge out: sort, drop the extreme `trim_fraction` at each end, take the
// midpoint of what's left.
double robustExtentMidpoint(std::vector<double> v, double trim_fraction)
{
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  const std::size_t trim = static_cast<std::size_t>(std::clamp(trim_fraction, 0.0, 0.49) * n);
  const double low = v[trim];
  const double high = v[n - 1 - trim];
  return 0.5 * (low + high);
}

}  // namespace

class ShoulderCenterlineNode : public rclcpp::Node
{
  using Image = sensor_msgs::msg::Image;
  using CameraInfo = sensor_msgs::msg::CameraInfo;
  using PointCloud2 = sensor_msgs::msg::PointCloud2;
  using Path = nav_msgs::msg::Path;
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image, PointCloud2>;

public:
  ShoulderCenterlineNode()
  : Node("shoulder_centerline_node")
  {
    image_topic_ = declare_parameter<std::string>("image_topic", "/sensing/camera/CAM_FRONT/image_raw");
    mask_topic_ = declare_parameter<std::string>("mask_topic", "/shoulder_detection_node/class_mask");
    camera_info_topic_ =
      declare_parameter<std::string>("camera_info_topic", "/sensing/camera/CAM_FRONT/camera_info");
    lidar_topic_ = declare_parameter<std::string>("lidar_topic", "/sensing/lidar/top/pointcloud_before_sync");
    output_frame_ = declare_parameter<std::string>("output_frame", "base_link");
    shoulder_class_ = declare_parameter<int>("shoulder_class", 1);
    sync_slop_ = declare_parameter<double>("sync_slop", 0.15);

    x_min_ = declare_parameter<double>("x_min", 3.0);
    x_max_ = declare_parameter<double>("x_max", 40.0);
    bin_size_ = declare_parameter<double>("bin_size", 1.0);
    min_points_per_bin_ = declare_parameter<int>("min_points_per_bin", 15);
    min_valid_bins_ = declare_parameter<int>("min_valid_bins", 4);
    z_min_ = declare_parameter<double>("z_min", -0.3);
    z_max_ = declare_parameter<double>("z_max", 0.5);

    // How each bin's lateral (y) center is computed from its raw painted
    // points. "median" is a density-weighted centroid -- biased toward
    // whichever edge of the shoulder happens to get more LiDAR hits, which is
    // scan-pattern/incidence-angle dependent, not the true lane center.
    // "extent_midpoint" (default) instead finds the left/right edge of the
    // classified points (robust/trimmed, see robustExtentMidpoint()) and
    // takes their midpoint -- the standard lane-centerline technique. Set
    // back to "median" (no rebuild needed) to instantly revert if this
    // causes a regression; see also the git checkpoint committed in this
    // workspace right before this parameter was introduced.
    centerline_estimator_ = declare_parameter<std::string>("centerline_estimator", "extent_midpoint");
    use_extent_midpoint_ = (centerline_estimator_ == "extent_midpoint");
    edge_trim_fraction_ = declare_parameter<double>("edge_trim_fraction", 0.1);

    // Where the lateral sample for each bin comes from in the first place.
    // "lidar": project LiDAR points into the mask, keep the ones landing on
    // the shoulder class -- fine on a wide shoulder, but on a narrow one
    // (e.g. a mountain-pass section with only ~1-1.5m of paved shoulder
    // before a guardrail) only a handful of sparse LiDAR rays land inside it
    // per frame, so *any* point-based estimate (median or extent-midpoint)
    // is working with too few, noisy samples -- that's what caused the
    // zigzag even after the extent-midpoint fix, not the aggregation method.
    // "mask_ipm" (default) instead works in the dense 2D mask: for each
    // sampled image row, find the *largest contiguous run* of shoulder-
    // classified columns (robust to disjoint noise elsewhere in the row),
    // take its midpoint column, and back-project that single pixel to 3D via
    // classical ground-plane inverse perspective mapping (ray-cast the pixel
    // through the known static camera pose, intersect the base_link z=0
    // plane) -- the same principle used by monocular lane-centering ADAS
    // systems. This has far higher effective resolution than sparse LiDAR
    // hits regardless of how narrow the real shoulder is, at the cost of
    // assuming locally flat ground (true for this project's documented flat
    // CARLA test routes). Set back to "lidar" (no rebuild) to revert.
    use_mask_center_ipm_ = declare_parameter<std::string>("bin_source", "mask_ipm") == "mask_ipm";
    mask_row_step_ = declare_parameter<int>("mask_row_step", 4);
    min_mask_run_px_ = declare_parameter<int>("min_mask_run_px", 3);

    waypoint_spacing_ = declare_parameter<double>("waypoint_spacing", 1.0);
    smoothing_alpha_ = declare_parameter<double>("smoothing_alpha", 0.3);

    path_topic_ = declare_parameter<std::string>("path_topic", "~/shoulder_centerline_path");
    debug_image_topic_ = declare_parameter<std::string>("debug_image_topic", "~/centerline_debug_image");
    publish_debug_image_ = declare_parameter<bool>("publish_debug_image", true);
    log_every_n_ = declare_parameter<int>("log_every_n", 30);

    // The path above is the *local* ego-relative window (base_link, re-fit
    // every frame -- intentionally forgets the past, since that's what a
    // local planner/controller wants). RViz's Path display always shows only
    // the latest message's poses, so viewing that alone looks like the line
    // "moves with the vehicle".
    //
    // This second, accumulating map-frame path is for visualization: every
    // banded LiDAR point (already computed in the main per-point loop below)
    // is *also* transformed into map frame and dropped into a persistent
    // world-frame bin, keyed by the vehicle's own cumulative traveled
    // distance plus that point's ego-relative forward offset at capture time
    // -- so the bin key is robust to route curvature without needing a
    // separate Frenet/heading reconstruction. Unlike the local path (one
    // noisy per-frame median per bin), a given real-world 1m road segment is
    // visible across the *entire* x_min..x_max window as the vehicle
    // approaches it -- roughly (x_max-x_min)/(distance travelled per frame)
    // repeat observations, often 50-100+ -- so each world bin accumulates
    // many independent samples over its visible lifetime before the vehicle
    // passes it, and the published path is the running median of all samples
    // collected so far per bin. This is the standard multi-sweep LiDAR
    // accumulation idea (as used in scan-accumulation/LiDAR mapping
    // pipelines): averaging many independent noisy observations is a
    // fundamentally more robust estimator than smoothing a single frame's
    // sparse fit, which is what the two earlier (tip-only-append, then
    // commit-lag) attempts were still doing and why they stayed jittery.
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    accumulate_path_ = declare_parameter<bool>("accumulate_path", true);
    accumulated_path_topic_ =
      declare_parameter<std::string>("accumulated_path_topic", "~/shoulder_centerline_path_map");
    // Minimum accumulated raw-point samples (across all frames) a world bin
    // needs before it's included in the published path -- higher trades a
    // longer lag behind the vehicle for a smoother median.
    min_world_bin_samples_ = declare_parameter<int>("min_world_bin_samples", 20);
    // Caps how many finalized world bins are retained/published (oldest dropped first).
    max_accumulated_points_ = declare_parameter<int>("max_accumulated_points", 2000);
    // Excludes the nearest N valid bins each frame from world-bin
    // accumulation only (never the local path). At short range the mask_ipm
    // ray is nearly vertical, so small mask-boundary pixel noise there maps
    // to a large swing in the back-projected ground point -- confirmed by
    // the user seeing exactly this as a curve at the start of the local path.
    world_bin_near_margin_ = declare_parameter<int>("world_bin_near_margin", 1);

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    camera_info_sub_ = create_subscription<CameraInfo>(
      camera_info_topic_, 1,
      [this](CameraInfo::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(camera_info_mutex_);
        latest_camera_info_ = msg;
      });

    path_pub_ = create_publisher<Path>(path_topic_, 1);
    accumulated_path_pub_ = create_publisher<Path>(accumulated_path_topic_, 1);
    debug_pub_ = create_publisher<Image>(debug_image_topic_, 1);

    image_sub_.subscribe(this, image_topic_, rmw_qos_profile_default);
    mask_sub_.subscribe(this, mask_topic_, rmw_qos_profile_default);
    lidar_sub_.subscribe(this, lidar_topic_, rmw_qos_profile_sensor_data);

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), image_sub_, mask_sub_, lidar_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(sync_slop_));
    sync_->registerCallback(
      std::bind(&ShoulderCenterlineNode::onSynced, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3));

    RCLCPP_INFO(get_logger(), "shoulder_centerline_node ready, waiting for camera_info + TF ...");
  }

private:
  std::optional<geometry_msgs::msg::TransformStamped> lookup(
    const std::string & target_frame, const std::string & source_frame)
  {
    try {
      return tf_buffer_->lookupTransform(target_frame, source_frame, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "TF lookup %s -> %s failed: %s",
        source_frame.c_str(), target_frame.c_str(), ex.what());
      return std::nullopt;
    }
  }

  double lateralCenter(const std::vector<double> & y) const
  {
    return use_extent_midpoint_ ? robustExtentMidpoint(y, edge_trim_fraction_) : median(y);
  }

  // For each sampled image row, finds the largest contiguous run of
  // shoulder-classified columns (robust to disjoint false-positive noise
  // elsewhere in the row -- a global min/max column would wrongly span
  // across two unrelated detections), takes its midpoint column, and
  // back-projects that single pixel to base_link via ground-plane (z=0)
  // inverse perspective mapping. Returns (x, y) points in base_link/
  // output_frame, ground assumption z=0.
  std::vector<Eigen::Vector2d> sampleMaskCenterline(
    const cv::Mat & mask, const Eigen::Isometry3d & t_base_cam, double fx, double fy, double cx, double cy) const
  {
    std::vector<Eigen::Vector2d> samples;
    const Eigen::Matrix3d r_base_cam = t_base_cam.rotation();
    const Eigen::Vector3d c_base = t_base_cam.translation();

    for (int row = 0; row < mask.rows; row += mask_row_step_) {
      const uint8_t * row_ptr = mask.ptr<uint8_t>(row);
      int run_start = -1, best_start = -1, best_len = 0;
      for (int col = 0; col <= mask.cols; ++col) {
        const bool on = (col < mask.cols) && (row_ptr[col] == shoulder_class_);
        if (on) {
          if (run_start < 0) {
            run_start = col;
          }
        } else if (run_start >= 0) {
          const int len = col - run_start;
          if (len > best_len) {
            best_len = len;
            best_start = run_start;
          }
          run_start = -1;
        }
      }
      if (best_len < min_mask_run_px_) {
        continue;
      }
      const double center_col = best_start + 0.5 * (best_len - 1);

      const Eigen::Vector3d d_cam((center_col - cx) / fx, (row - cy) / fy, 1.0);
      const Eigen::Vector3d d_base = r_base_cam * d_cam;
      if (d_base.z() >= -1e-6) {
        continue;  // ray parallel to or pointing above the ground plane -- no valid intersection ahead
      }
      const double t = -c_base.z() / d_base.z();
      if (t <= 0.0) {
        continue;  // intersection behind the camera
      }
      const Eigen::Vector3d p = c_base + t * d_base;
      if (p.x() < x_min_ || p.x() > x_max_) {
        continue;
      }
      samples.emplace_back(p.x(), p.y());
    }
    return samples;
  }

  void onSynced(
    Image::ConstSharedPtr image_msg, Image::ConstSharedPtr mask_msg, PointCloud2::ConstSharedPtr lidar_msg)
  {
    const auto t0 = std::chrono::steady_clock::now();

    CameraInfo::ConstSharedPtr camera_info;
    {
      std::lock_guard<std::mutex> lock(camera_info_mutex_);
      camera_info = latest_camera_info_;
    }
    if (!camera_info) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "no camera_info received yet, skipping frame");
      return;
    }

    const std::string cam_frame =
      !mask_msg->header.frame_id.empty() ? mask_msg->header.frame_id : image_msg->header.frame_id;
    const std::string lidar_frame = lidar_msg->header.frame_id;

    const auto t_cam_lidar_msg = lookup(cam_frame, lidar_frame);
    const auto t_base_lidar_msg = lookup(output_frame_, lidar_frame);
    if (!t_cam_lidar_msg || !t_base_lidar_msg) {
      return;
    }
    const Eigen::Isometry3d t_cam_lidar = tf2::transformToEigen(*t_cam_lidar_msg);
    const Eigen::Isometry3d t_base_lidar = tf2::transformToEigen(*t_base_lidar_msg);

    // For the accumulated map-frame path: look up map<-base_link once per
    // frame (used to project banded points into world bins below) and update
    // the vehicle's own cumulative traveled distance, which is what keys
    // those world bins (robust to route curvature, no heading/Frenet
    // reconstruction needed).
    std::optional<Eigen::Isometry3d> t_map_base;
    if (accumulate_path_) {
      if (const auto t_map_base_msg = lookup(map_frame_, output_frame_)) {
        t_map_base = tf2::transformToEigen(*t_map_base_msg);
        const Eigen::Vector2d ego_xy = t_map_base->translation().head<2>();
        if (have_ego_origin_) {
          traveled_arclength_ += (ego_xy - prev_ego_xy_).norm();
        }
        prev_ego_xy_ = ego_xy;
        have_ego_origin_ = true;
      }
    }

    cv_bridge::CvImageConstPtr mask_cv = cv_bridge::toCvShare(mask_msg, "mono8");
    const cv::Mat & mask = mask_cv->image;
    const int w = mask.cols, h = mask.rows;

    const double fx = camera_info->k[0], fy = camera_info->k[4];
    const double cx = camera_info->k[2], cy = camera_info->k[5];

    const int num_bins = static_cast<int>(std::ceil((x_max_ - x_min_) / bin_size_));
    std::vector<std::vector<double>> bin_y(num_bins), bin_z(num_bins);
    // Raw-x/map-frame-point pairs per bin, collected but not yet pushed into
    // world_bins_ -- deferred until after we know which bins are the
    // frame's nearest valid ones (see world_bin_near_margin_ below), so
    // those can be excluded from the accumulation without touching the
    // local fit at all.
    std::vector<std::vector<std::pair<double, Eigen::Vector3d>>> bin_world_candidates(num_bins);
    std::vector<cv::Point> painted_px;

    std::size_t total_points = 0, in_bounds_points = 0, painted_points = 0, banded_points = 0;
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*lidar_msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*lidar_msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*lidar_msg, "z");
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      ++total_points;
      if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y) || !std::isfinite(*iter_z)) {
        continue;
      }
      const Eigen::Vector3d p_lidar(*iter_x, *iter_y, *iter_z);

      const Eigen::Vector3d p_cam = t_cam_lidar * p_lidar;
      if (p_cam.z() <= 0.1) {
        continue;
      }
      const int u = static_cast<int>(fx * p_cam.x() / p_cam.z() + cx);
      const int v = static_cast<int>(fy * p_cam.y() / p_cam.z() + cy);
      if (u < 0 || u >= w || v < 0 || v >= h) {
        continue;
      }
      ++in_bounds_points;
      if (mask.at<uint8_t>(v, u) != shoulder_class_) {
        continue;
      }
      ++painted_points;
      painted_px.emplace_back(u, v);

      const Eigen::Vector3d p_base = t_base_lidar * p_lidar;
      if (p_base.z() < z_min_ || p_base.z() > z_max_) {
        continue;
      }
      if (p_base.x() < x_min_ || p_base.x() > x_max_) {
        continue;
      }
      ++banded_points;

      // When bin_source=mask_ipm (default), the LiDAR loop above still runs
      // in full -- it's what produces painted_px and the diagnostic counts
      // logged below -- but the bins themselves are populated separately,
      // from the dense mask-boundary IPM samples (see below), not from these
      // sparse LiDAR hits.
      if (!use_mask_center_ipm_) {
        const int b = static_cast<int>((p_base.x() - x_min_) / bin_size_);
        if (b >= 0 && b < num_bins) {
          bin_y[b].push_back(p_base.y());
          bin_z[b].push_back(p_base.z());
          if (t_map_base) {
            bin_world_candidates[b].emplace_back(p_base.x(), *t_map_base * p_base);
          }
        }
      }
    }

    if (use_mask_center_ipm_) {
      if (const auto t_base_cam_msg = lookup(output_frame_, cam_frame)) {
        const Eigen::Isometry3d t_base_cam = tf2::transformToEigen(*t_base_cam_msg);
        for (const auto & s : sampleMaskCenterline(mask, t_base_cam, fx, fy, cx, cy)) {
          const int b = static_cast<int>((s.x() - x_min_) / bin_size_);
          if (b >= 0 && b < num_bins) {
            bin_y[b].push_back(s.y());
            bin_z[b].push_back(0.0);
            if (t_map_base) {
              bin_world_candidates[b].emplace_back(s.x(), *t_map_base * Eigen::Vector3d(s.x(), s.y(), 0.0));
            }
          }
        }
      }
    }

    std::vector<double> cx_list, cy_list, cz_list;
    std::vector<int> valid_bin_idx;
    int valid_bins = 0;
    for (int b = 0; b < num_bins; ++b) {
      if (static_cast<int>(bin_y[b].size()) < min_points_per_bin_) {
        continue;
      }
      ++valid_bins;
      valid_bin_idx.push_back(b);
      cx_list.push_back(x_min_ + (b + 0.5) * bin_size_);
      cy_list.push_back(lateralCenter(bin_y[b]));
      cz_list.push_back(median(bin_z[b]));
    }

    // The nearest bin(s) each frame can be unreliable: at short range the
    // mask_ipm ray is nearly vertical, so small mask-boundary pixel noise
    // there maps to a large swing in the back-projected ground point --
    // directly confirmed by the user seeing a curve at the *start* of the
    // local path. Excluding world_bin_near_margin_ nearest valid bins from
    // world accumulation only (never the local path, which already looks
    // right on its own) keeps that curve from being baked into the
    // permanent trail. Deliberately NOT also excluding the farthest bins
    // (an earlier version of this did) -- the user asked for that reverted,
    // since 58183c7's original (unrestricted) far end already looked right.
    std::vector<bool> bin_world_eligible(num_bins, false);
    {
      const int near_cutoff = std::min(static_cast<int>(valid_bin_idx.size()), world_bin_near_margin_);
      for (int i = near_cutoff; i < static_cast<int>(valid_bin_idx.size()); ++i) {
        bin_world_eligible[valid_bin_idx[i]] = true;
      }
    }

    std::vector<Waypoint> waypoints;
    if (valid_bins >= min_valid_bins_) {
      waypoints = fitCenterline(cx_list, cy_list, cz_list);
    }

    if (!waypoints.empty()) {
      publishPath(waypoints, mask_msg->header.stamp);
    }

    // Independent of the local per-frame fit above -- the world bins are
    // populated from the same eligible-bin candidates collected in the main
    // loop, so this publishes/refines even on frames where the local fit
    // didn't converge (e.g. too few bins for min_valid_bins_).
    if (accumulate_path_ && t_map_base) {
      for (int b = 0; b < num_bins; ++b) {
        if (!bin_world_eligible[b]) {
          continue;
        }
        for (const auto & [raw_x, p_map] : bin_world_candidates[b]) {
          const long long world_key = static_cast<long long>(std::floor((traveled_arclength_ + raw_x) / bin_size_));
          auto & wb = world_bins_[world_key];
          wb.x.push_back(p_map.x());
          wb.y.push_back(p_map.y());
          wb.z.push_back(p_map.z());
        }
      }
      publishAccumulatedPath(mask_msg->header.stamp);
    }

    if (publish_debug_image_) {
      publishDebugImage(
        image_msg, painted_px, waypoints, t_cam_lidar, t_base_lidar, fx, fy, cx, cy, valid_bins, num_bins);
    }

    const double dt_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (frame_idx_ % static_cast<std::size_t>(std::max(1, log_every_n_)) == 0) {
      RCLCPP_INFO(
        get_logger(),
        "frame %6zu | %5.1f ms | lidar %zu -> in_img %zu -> painted %zu -> banded %zu | bins %d/%d | "
        "waypoints %zu",
        frame_idx_, dt_ms, total_points, in_bounds_points, painted_points, banded_points, valid_bins, num_bins,
        waypoints.size());
    }
    ++frame_idx_;
  }

  std::vector<Waypoint> fitCenterline(
    const std::vector<double> & xs, const std::vector<double> & ys, const std::vector<double> & zs)
  {
    // Uniform-knot parametrization over the *sequence* of valid bins (t=0..n-1),
    // not chord length -- a documented simplification vs. a chord-length spline:
    // accurate as long as valid bins are mostly contiguous (the common case),
    // slightly under/over-weights segments following a gap of skipped bins.
    const int n = static_cast<int>(xs.size());
    if (n < 2) {
      return {};  // need at least 2 knots for a heading/arclength to exist
    }
    shoulder_centerline::CubicSpline1D spline_x(xs);
    shoulder_centerline::CubicSpline1D spline_y(ys);
    shoulder_centerline::CubicSpline1D spline_z(zs);
    constexpr int kSamplesPerSegment = 20;
    const int dense_n = (n - 1) * kSamplesPerSegment + 1;
    std::vector<double> t_dense(dense_n), arclen(dense_n);
    arclen[0] = 0.0;
    double prev_x = spline_x.eval(0.0), prev_y = spline_y.eval(0.0);
    t_dense[0] = 0.0;
    for (int i = 1; i < dense_n; ++i) {
      const double t = static_cast<double>(i) * (n - 1) / (dense_n - 1);
      t_dense[i] = t;
      const double px = spline_x.eval(t), py = spline_y.eval(t);
      arclen[i] = arclen[i - 1] + std::hypot(px - prev_x, py - prev_y);
      prev_x = px;
      prev_y = py;
    }
    const double total_len = arclen.back();
    if (total_len < waypoint_spacing_) {
      return {};
    }

    const int num_steps = static_cast<int>(total_len / waypoint_spacing_) + 1;
    std::vector<Waypoint> waypoints;
    waypoints.reserve(num_steps);

    for (int k = 0; k < num_steps; ++k) {
      const double target_s = k * waypoint_spacing_;
      const auto it = std::lower_bound(arclen.begin(), arclen.end(), target_s);
      double t;
      if (it == arclen.begin()) {
        t = t_dense.front();
      } else if (it == arclen.end()) {
        t = t_dense.back();
      } else {
        const std::size_t idx = std::distance(arclen.begin(), it);
        const double s0 = arclen[idx - 1], s1 = arclen[idx];
        const double frac = (s1 > s0) ? (target_s - s0) / (s1 - s0) : 0.0;
        t = t_dense[idx - 1] + frac * (t_dense[idx] - t_dense[idx - 1]);
      }

      const double x = spline_x.eval(t), y = spline_y.eval(t), z = spline_z.eval(t);
      const double xp = spline_x.eval(t, 1), yp = spline_y.eval(t, 1);
      const double xpp = spline_x.eval(t, 2), ypp = spline_y.eval(t, 2);
      const double denom = std::pow(xp * xp + yp * yp, 1.5);
      const double kappa = (denom > 1e-9) ? (xp * ypp - yp * xpp) / denom : 0.0;
      const double yaw = std::atan2(yp, xp);

      auto smoothed_it = smoothed_y_by_step_.find(k);
      double y_smoothed = y;
      if (smoothed_it != smoothed_y_by_step_.end()) {
        y_smoothed = smoothing_alpha_ * y + (1.0 - smoothing_alpha_) * smoothed_it->second;
      }
      smoothed_y_by_step_[k] = y_smoothed;

      waypoints.push_back({x, y_smoothed, z, yaw, kappa});
    }

    for (auto it = smoothed_y_by_step_.begin(); it != smoothed_y_by_step_.end();) {
      if (it->first >= num_steps) {
        it = smoothed_y_by_step_.erase(it);
      } else {
        ++it;
      }
    }

    return waypoints;
  }

  void publishPath(const std::vector<Waypoint> & waypoints, const builtin_interfaces::msg::Time & stamp)
  {
    Path path;
    path.header.frame_id = output_frame_;
    path.header.stamp = stamp;
    path.poses.reserve(waypoints.size());
    for (const auto & wp : waypoints) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = wp.x;
      pose.pose.position.y = wp.y;
      pose.pose.position.z = wp.z;
      pose.pose.orientation.z = std::sin(wp.yaw / 2.0);
      pose.pose.orientation.w = std::cos(wp.yaw / 2.0);
      path.poses.push_back(pose);
    }
    path_pub_->publish(path);
  }

  // Builds/publishes the accumulated path directly from world_bins_ (already
  // populated in onSynced's main per-point loop). Each bin's published
  // position is the running median of *every* sample ever collected for it,
  // not a single frame's fit -- see the module-level comment on
  // min_world_bin_samples_ for why this is the actual fix for jitter, not
  // just another smoothing-parameter tweak.
  void publishAccumulatedPath(const builtin_interfaces::msg::Time & stamp)
  {
    // Prune oldest (smallest key = farthest behind the vehicle) bins first,
    // same eviction order the deque-based version used.
    while (static_cast<int>(world_bins_.size()) > max_accumulated_points_) {
      world_bins_.erase(world_bins_.begin());
    }

    Path path;
    path.header.frame_id = map_frame_;
    path.header.stamp = stamp;

    std::vector<Eigen::Vector3d> points;
    points.reserve(world_bins_.size());
    for (const auto & [key, wb] : world_bins_) {
      if (static_cast<int>(wb.x.size()) < min_world_bin_samples_) {
        continue;
      }
      points.emplace_back(median(wb.x), lateralCenter(wb.y), median(wb.z));
    }

    path.poses.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
      const auto & p = points[i];
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = p.x();
      pose.pose.position.y = p.y();
      pose.pose.position.z = p.z();
      double yaw = 0.0;
      if (i > 0) {
        const auto & prev = points[i - 1];
        yaw = std::atan2(p.y() - prev.y(), p.x() - prev.x());
      }
      pose.pose.orientation.z = std::sin(yaw / 2.0);
      pose.pose.orientation.w = std::cos(yaw / 2.0);
      path.poses.push_back(pose);
    }
    accumulated_path_pub_->publish(path);
  }

  void publishDebugImage(
    Image::ConstSharedPtr image_msg, const std::vector<cv::Point> & painted_px,
    const std::vector<Waypoint> & waypoints, const Eigen::Isometry3d & t_cam_lidar,
    const Eigen::Isometry3d & t_base_lidar, double fx, double fy, double cx, double cy, int valid_bins,
    int num_bins)
  {
    cv_bridge::CvImageConstPtr img_cv = cv_bridge::toCvShare(image_msg, "bgr8");
    cv::Mat frame = img_cv->image.clone();

    for (const auto & px : painted_px) {
      cv::circle(frame, px, 2, cv::Scalar(0, 255, 255), -1);
    }

    if (!waypoints.empty()) {
      const Eigen::Isometry3d t_cam_base = t_cam_lidar * t_base_lidar.inverse();
      bool have_prev = false;
      cv::Point prev_pt;
      for (const auto & wp : waypoints) {
        const Eigen::Vector3d p_cam = t_cam_base * Eigen::Vector3d(wp.x, wp.y, wp.z);
        if (p_cam.z() <= 0.1) {
          have_prev = false;
          continue;
        }
        const cv::Point px(
          static_cast<int>(fx * p_cam.x() / p_cam.z() + cx), static_cast<int>(fy * p_cam.y() / p_cam.z() + cy));
        cv::circle(frame, px, 3, cv::Scalar(255, 0, 255), -1);
        if (have_prev) {
          cv::line(frame, prev_pt, px, cv::Scalar(255, 0, 255), 2);
        }
        prev_pt = px;
        have_prev = true;
      }
    }

    char hud[128];
    std::snprintf(
      hud, sizeof(hud), "bins %d/%d | waypoints %zu", valid_bins, num_bins, waypoints.size());
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, 26), cv::Scalar(0, 0, 0), -1);
    cv::putText(
      frame, hud, cv::Point(8, 18), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    cv_bridge::CvImage out;
    out.header = image_msg->header;
    out.encoding = "bgr8";
    out.image = frame;
    debug_pub_->publish(*out.toImageMsg());
  }

  // Params
  std::string image_topic_, mask_topic_, camera_info_topic_, lidar_topic_, output_frame_;
  std::string path_topic_, debug_image_topic_;
  int shoulder_class_;
  double sync_slop_;
  double x_min_, x_max_, bin_size_;
  int min_points_per_bin_, min_valid_bins_;
  double z_min_, z_max_;
  std::string centerline_estimator_;
  bool use_extent_midpoint_;
  double edge_trim_fraction_;
  bool use_mask_center_ipm_;
  int mask_row_step_;
  int min_mask_run_px_;
  double waypoint_spacing_, smoothing_alpha_;
  bool publish_debug_image_;
  int log_every_n_;

  std::string map_frame_;
  bool accumulate_path_;
  int max_accumulated_points_;
  std::string accumulated_path_topic_;
  int min_world_bin_samples_;
  int world_bin_near_margin_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<CameraInfo>::SharedPtr camera_info_sub_;
  CameraInfo::ConstSharedPtr latest_camera_info_;
  std::mutex camera_info_mutex_;

  rclcpp::Publisher<Path>::SharedPtr path_pub_;
  rclcpp::Publisher<Path>::SharedPtr accumulated_path_pub_;
  rclcpp::Publisher<Image>::SharedPtr debug_pub_;

  message_filters::Subscriber<Image> image_sub_, mask_sub_;
  message_filters::Subscriber<PointCloud2> lidar_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  std::map<int, double> smoothed_y_by_step_;
  std::map<long long, WorldBinSamples> world_bins_;
  bool have_ego_origin_ = false;
  Eigen::Vector2d prev_ego_xy_ = Eigen::Vector2d::Zero();
  double traveled_arclength_ = 0.0;
  std::size_t frame_idx_ = 0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ShoulderCenterlineNode>());
  rclcpp::shutdown();
  return 0;
}
