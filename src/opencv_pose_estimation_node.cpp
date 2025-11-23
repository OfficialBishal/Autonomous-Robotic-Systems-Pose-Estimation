#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <yaml-cpp/yaml.h>

class OpenCVPoseEstimation {
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    
    message_filters::Subscriber<sensor_msgs::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo> SyncPolicy;
    message_filters::Synchronizer<SyncPolicy> sync_;
    
    ros::Publisher pose_pub_;
    image_transport::Publisher vis_image_pub_;
    
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    cv::Ptr<cv::ORB> orb_detector_;
    cv::FlannBasedMatcher matcher_;
    
    std::vector<cv::Point3f> model_points_;
    std::vector<cv::Mat> model_descriptors_;
    
    cv::KalmanFilter kf_;
    bool kalman_initialized_;
    int min_inliers_kalman_;
    
    int num_keypoints_;
    float ratio_test_;
    int ransac_iterations_;
    double ransac_reprojection_error_;
    double ransac_confidence_;
    std::string model_file_path_;
    std::string image_topic_;
    std::string camera_info_topic_;
    std::string frame_id_;
    
    bool loadModel(const std::string& model_path);
    cv::Mat rot2euler(const cv::Mat& rotation_matrix);
    cv::Mat euler2rot(const cv::Mat& euler);
    void initKalmanFilter();
    void fillMeasurements(cv::Mat& measurements, const cv::Mat& translation, const cv::Mat& rotation);
    void updateKalmanFilter(cv::Mat& measurements, cv::Mat& translation_estimated, cv::Mat& rotation_estimated);
    
public:
    OpenCVPoseEstimation();
    void imageCallback(const sensor_msgs::ImageConstPtr& image_msg, const sensor_msgs::CameraInfoConstPtr& info_msg);
};

OpenCVPoseEstimation::OpenCVPoseEstimation() 
    : nh_("~"), it_(nh_), 
      image_sub_(nh_, "", 1),
      info_sub_(nh_, "", 1),
      sync_(SyncPolicy(10), image_sub_, info_sub_),
      kalman_initialized_(false)
{
    nh_.param("num_keypoints", num_keypoints_, 2000);
    nh_.param("ratio_test", ratio_test_, 0.7f);
    nh_.param("ransac_iterations", ransac_iterations_, 500);
    nh_.param("ransac_reprojection_error", ransac_reprojection_error_, 2.0);
    nh_.param("ransac_confidence", ransac_confidence_, 0.95);
    nh_.param("min_inliers_kalman", min_inliers_kalman_, 30);
    nh_.param("model_file", model_file_path_, std::string(""));
    nh_.param("image_topic", image_topic_, std::string("/hsrb/head_rgbd_sensor/rgb/image_rect_color"));
    nh_.param("camera_info_topic", camera_info_topic_, std::string("/hsrb/head_rgbd_sensor/rgb/camera_info"));
    nh_.param("frame_id", frame_id_, std::string("head_rgbd_sensor_rgb_frame"));
    
    orb_detector_ = cv::ORB::create(num_keypoints_);
    matcher_ = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1));
    
    if (model_file_path_.empty()) {
        ROS_ERROR("Model file path not specified! Please set the 'model_file' parameter.");
        return;
    }
    
    if (!loadModel(model_file_path_)) {
        ROS_ERROR("Failed to load 3D model from: %s", model_file_path_.c_str());
        return;
    }
    
    ROS_INFO("Loaded %zu 3D model points", model_points_.size());
    
    initKalmanFilter();
    
    image_sub_.subscribe(nh_, image_topic_, 1);
    info_sub_.subscribe(nh_, camera_info_topic_, 1);
    sync_.registerCallback(boost::bind(&OpenCVPoseEstimation::imageCallback, this, _1, _2));
    
    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("pose", 10);
    vis_image_pub_ = it_.advertise("visualization_image", 1);
    
    ROS_INFO("OpenCV Pose Estimation node initialized");
    ROS_INFO("Subscribing to image topic: %s", image_topic_.c_str());
    ROS_INFO("Subscribing to camera_info topic: %s", camera_info_topic_.c_str());
}

bool OpenCVPoseEstimation::loadModel(const std::string& model_path) {
    try {
        YAML::Node config = YAML::LoadFile(model_path);
        if (!config["points"]) {
            ROS_ERROR("Model file missing 'points' section");
            return false;
        }
        
        model_points_.clear();
        model_descriptors_.clear();
        
        for (const auto& point_node : config["points"]) {
            cv::Point3f pt;
            pt.x = point_node["x"].as<float>();
            pt.y = point_node["y"].as<float>();
            pt.z = point_node["z"].as<float>();
            model_points_.push_back(pt);
            
            if (point_node["descriptor"]) {
                std::vector<float> desc_vec = point_node["descriptor"].as<std::vector<float>>();
                // ORB descriptors are binary (0-255), convert to CV_8U
                cv::Mat desc(1, desc_vec.size(), CV_8U);
                for (size_t i = 0; i < desc_vec.size(); i++) {
                    desc.at<uchar>(0, i) = static_cast<uchar>(std::max(0.0f, std::min(255.0f, desc_vec[i])));
                }
                model_descriptors_.push_back(desc);
            }
        }
        
        if (!model_descriptors_.empty()) {
            ROS_INFO("Loaded %zu descriptors", model_descriptors_.size());
        }
        
        return true;
    } catch (const std::exception& e) {
        ROS_ERROR("Error loading model file: %s", e.what());
        return false;
    }
}

void OpenCVPoseEstimation::initKalmanFilter() {
    kf_.init(18, 6, 0, CV_64F);
    cv::setIdentity(kf_.transitionMatrix);
    double dt = 0.033;
    kf_.transitionMatrix.at<double>(0, 3) = dt;
    kf_.transitionMatrix.at<double>(1, 4) = dt;
    kf_.transitionMatrix.at<double>(2, 5) = dt;
    kf_.transitionMatrix.at<double>(3, 6) = dt;
    kf_.transitionMatrix.at<double>(4, 7) = dt;
    kf_.transitionMatrix.at<double>(5, 8) = dt;
    kf_.transitionMatrix.at<double>(9, 12) = dt;
    kf_.transitionMatrix.at<double>(10, 13) = dt;
    kf_.transitionMatrix.at<double>(11, 14) = dt;
    kf_.transitionMatrix.at<double>(12, 15) = dt;
    kf_.transitionMatrix.at<double>(13, 16) = dt;
    kf_.transitionMatrix.at<double>(14, 17) = dt;
    
    cv::setIdentity(kf_.measurementMatrix);
    kf_.measurementMatrix.at<double>(0, 0) = 1;
    kf_.measurementMatrix.at<double>(1, 1) = 1;
    kf_.measurementMatrix.at<double>(2, 2) = 1;
    kf_.measurementMatrix.at<double>(3, 9) = 1;
    kf_.measurementMatrix.at<double>(4, 10) = 1;
    kf_.measurementMatrix.at<double>(5, 11) = 1;
    
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-5));
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));
    
    kalman_initialized_ = true;
}

cv::Mat OpenCVPoseEstimation::rot2euler(const cv::Mat& rotation_matrix) {
    cv::Mat euler(3, 1, CV_64F);
    double sy = sqrt(rotation_matrix.at<double>(0, 0) * rotation_matrix.at<double>(0, 0) + 
                     rotation_matrix.at<double>(1, 0) * rotation_matrix.at<double>(1, 0));
    bool singular = sy < 1e-6;
    
    if (!singular) {
        euler.at<double>(0) = atan2(rotation_matrix.at<double>(2, 1), rotation_matrix.at<double>(2, 2));
        euler.at<double>(1) = atan2(-rotation_matrix.at<double>(2, 0), sy);
        euler.at<double>(2) = atan2(rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(0, 0));
    } else {
        euler.at<double>(0) = atan2(-rotation_matrix.at<double>(1, 2), rotation_matrix.at<double>(1, 1));
        euler.at<double>(1) = atan2(-rotation_matrix.at<double>(2, 0), sy);
        euler.at<double>(2) = 0;
    }
    return euler;
}

cv::Mat OpenCVPoseEstimation::euler2rot(const cv::Mat& euler) {
    cv::Mat R_x = (cv::Mat_<double>(3, 3) <<
        1, 0, 0,
        0, cos(euler.at<double>(0)), -sin(euler.at<double>(0)),
        0, sin(euler.at<double>(0)), cos(euler.at<double>(0)));
    
    cv::Mat R_y = (cv::Mat_<double>(3, 3) <<
        cos(euler.at<double>(1)), 0, sin(euler.at<double>(1)),
        0, 1, 0,
        -sin(euler.at<double>(1)), 0, cos(euler.at<double>(1)));
    
    cv::Mat R_z = (cv::Mat_<double>(3, 3) <<
        cos(euler.at<double>(2)), -sin(euler.at<double>(2)), 0,
        sin(euler.at<double>(2)), cos(euler.at<double>(2)), 0,
        0, 0, 1);
    
    return R_z * R_y * R_x;
}

void OpenCVPoseEstimation::fillMeasurements(cv::Mat& measurements, const cv::Mat& translation, const cv::Mat& rotation) {
    cv::Mat measured_eulers = rot2euler(rotation);
    measurements.at<double>(0) = translation.at<double>(0);
    measurements.at<double>(1) = translation.at<double>(1);
    measurements.at<double>(2) = translation.at<double>(2);
    measurements.at<double>(3) = measured_eulers.at<double>(0);
    measurements.at<double>(4) = measured_eulers.at<double>(1);
    measurements.at<double>(5) = measured_eulers.at<double>(2);
}

void OpenCVPoseEstimation::updateKalmanFilter(cv::Mat& measurements, cv::Mat& translation_estimated, cv::Mat& rotation_estimated) {
    cv::Mat prediction = kf_.predict();
    cv::Mat estimated = kf_.correct(measurements);
    
    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    translation_estimated.at<double>(2) = estimated.at<double>(2);
    
    cv::Mat eulers_estimated(3, 1, CV_64F);
    eulers_estimated.at<double>(0) = estimated.at<double>(9);
    eulers_estimated.at<double>(1) = estimated.at<double>(10);
    eulers_estimated.at<double>(2) = estimated.at<double>(11);
    
    rotation_estimated = euler2rot(eulers_estimated);
}

void OpenCVPoseEstimation::imageCallback(const sensor_msgs::ImageConstPtr& image_msg, 
                                         const sensor_msgs::CameraInfoConstPtr& info_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
    cv::Mat frame = cv_ptr->image;
    cv::Mat frame_vis = frame.clone();
    
    camera_matrix_ = cv::Mat(3, 3, CV_64F);
    camera_matrix_.at<double>(0, 0) = info_msg->K[0];
    camera_matrix_.at<double>(0, 1) = info_msg->K[1];
    camera_matrix_.at<double>(0, 2) = info_msg->K[2];
    camera_matrix_.at<double>(1, 0) = info_msg->K[3];
    camera_matrix_.at<double>(1, 1) = info_msg->K[4];
    camera_matrix_.at<double>(1, 2) = info_msg->K[5];
    camera_matrix_.at<double>(2, 0) = info_msg->K[6];
    camera_matrix_.at<double>(2, 1) = info_msg->K[7];
    camera_matrix_.at<double>(2, 2) = info_msg->K[8];
    
    dist_coeffs_ = cv::Mat(info_msg->D.size(), 1, CV_64F);
    for (size_t i = 0; i < info_msg->D.size(); i++) {
        dist_coeffs_.at<double>(i) = info_msg->D[i];
    }
    
    std::vector<cv::KeyPoint> scene_keypoints;
    cv::Mat scene_descriptors;
    orb_detector_->detectAndCompute(frame, cv::noArray(), scene_keypoints, scene_descriptors);
    
    if (scene_descriptors.empty()) {
        ROS_DEBUG_THROTTLE(5.0, "No scene descriptors found in image");
        sensor_msgs::ImagePtr vis_msg = cv_bridge::CvImage(image_msg->header, "bgr8", frame_vis).toImageMsg();
        vis_image_pub_.publish(vis_msg);
        return;
    }
    
    if (model_descriptors_.empty()) {
        ROS_WARN_THROTTLE(5.0, "No model descriptors loaded. Please provide a valid model file.");
        sensor_msgs::ImagePtr vis_msg = cv_bridge::CvImage(image_msg->header, "bgr8", frame_vis).toImageMsg();
        vis_image_pub_.publish(vis_msg);
        return;
    }
    
    // Match descriptors
    std::vector<std::vector<cv::DMatch>> knn_matches;
    try {
        if (model_descriptors_.size() > 0) {
            cv::Mat model_descriptors_mat;
            std::vector<cv::Mat> desc_list;
            for (const auto& desc : model_descriptors_) {
                desc_list.push_back(desc);
            }
            cv::vconcat(desc_list, model_descriptors_mat);
            
            // Convert descriptors to proper format if needed (ORB uses binary descriptors)
            // ORB descriptors are typically CV_8U, but we stored them as CV_32F in YAML
            // Convert back to CV_8U for proper matching
            cv::Mat model_descriptors_uint8;
            if (model_descriptors_mat.type() == CV_32F) {
                model_descriptors_mat.convertTo(model_descriptors_uint8, CV_8U);
            } else {
                model_descriptors_uint8 = model_descriptors_mat;
            }
            
            cv::Mat scene_descriptors_uint8;
            if (scene_descriptors.type() == CV_32F) {
                scene_descriptors.convertTo(scene_descriptors_uint8, CV_8U);
            } else {
                scene_descriptors_uint8 = scene_descriptors;
            }
            
            matcher_.knnMatch(scene_descriptors_uint8, model_descriptors_uint8, knn_matches, 2);
        }
    } catch (const cv::Exception& e) {
        ROS_DEBUG_THROTTLE(5.0, "Matching error: %s", e.what());
        sensor_msgs::ImagePtr vis_msg = cv_bridge::CvImage(image_msg->header, "bgr8", frame_vis).toImageMsg();
        vis_image_pub_.publish(vis_msg);
        return;
    }
    
    // Apply ratio test
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point3f> matched_model_points;
    std::vector<cv::Point2f> matched_scene_points;
    
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() == 2) {
            if (knn_matches[i][0].distance < ratio_test_ * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
                if (knn_matches[i][0].trainIdx < (int)model_points_.size()) {
                    matched_model_points.push_back(model_points_[knn_matches[i][0].trainIdx]);
                    matched_scene_points.push_back(scene_keypoints[knn_matches[i][0].queryIdx].pt);
                }
            }
        }
    }
    
    ROS_DEBUG("Found %zu good matches out of %zu total matches", good_matches.size(), knn_matches.size());
    
    if (matched_model_points.size() >= 4 && matched_scene_points.size() >= 4) {
        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        
        cv::solvePnPRansac(matched_model_points, matched_scene_points, 
                          camera_matrix_, dist_coeffs_,
                          rvec, tvec, false,
                          ransac_iterations_, ransac_reprojection_error_,
                          ransac_confidence_, inliers, cv::SOLVEPNP_ITERATIVE);
        
        if (inliers.size() >= min_inliers_kalman_) {
            cv::Mat rotation_matrix;
            cv::Rodrigues(rvec, rotation_matrix);
            
            cv::Mat measurements(6, 1, CV_64F);
            fillMeasurements(measurements, tvec, rotation_matrix);
            
            cv::Mat translation_estimated(3, 1, CV_64F);
            cv::Mat rotation_estimated(3, 3, CV_64F);
            updateKalmanFilter(measurements, translation_estimated, rotation_estimated);
            
            double trace = rotation_estimated.at<double>(0, 0) + rotation_estimated.at<double>(1, 1) + rotation_estimated.at<double>(2, 2);
            double qw, qx, qy, qz;
            if (trace > 0) {
                double s = sqrt(trace + 1.0) * 2;
                qw = 0.25 * s;
                qx = (rotation_estimated.at<double>(2, 1) - rotation_estimated.at<double>(1, 2)) / s;
                qy = (rotation_estimated.at<double>(0, 2) - rotation_estimated.at<double>(2, 0)) / s;
                qz = (rotation_estimated.at<double>(1, 0) - rotation_estimated.at<double>(0, 1)) / s;
            } else {
                if (rotation_estimated.at<double>(0, 0) > rotation_estimated.at<double>(1, 1) && 
                    rotation_estimated.at<double>(0, 0) > rotation_estimated.at<double>(2, 2)) {
                    double s = sqrt(1.0 + rotation_estimated.at<double>(0, 0) - rotation_estimated.at<double>(1, 1) - rotation_estimated.at<double>(2, 2)) * 2;
                    qw = (rotation_estimated.at<double>(2, 1) - rotation_estimated.at<double>(1, 2)) / s;
                    qx = 0.25 * s;
                    qy = (rotation_estimated.at<double>(0, 1) + rotation_estimated.at<double>(1, 0)) / s;
                    qz = (rotation_estimated.at<double>(0, 2) + rotation_estimated.at<double>(2, 0)) / s;
                } else if (rotation_estimated.at<double>(1, 1) > rotation_estimated.at<double>(2, 2)) {
                    double s = sqrt(1.0 + rotation_estimated.at<double>(1, 1) - rotation_estimated.at<double>(0, 0) - rotation_estimated.at<double>(2, 2)) * 2;
                    qw = (rotation_estimated.at<double>(0, 2) - rotation_estimated.at<double>(2, 0)) / s;
                    qx = (rotation_estimated.at<double>(0, 1) + rotation_estimated.at<double>(1, 0)) / s;
                    qy = 0.25 * s;
                    qz = (rotation_estimated.at<double>(1, 2) + rotation_estimated.at<double>(2, 1)) / s;
                } else {
                    double s = sqrt(1.0 + rotation_estimated.at<double>(2, 2) - rotation_estimated.at<double>(0, 0) - rotation_estimated.at<double>(1, 1)) * 2;
                    qw = (rotation_estimated.at<double>(1, 0) - rotation_estimated.at<double>(0, 1)) / s;
                    qx = (rotation_estimated.at<double>(0, 2) + rotation_estimated.at<double>(2, 0)) / s;
                    qy = (rotation_estimated.at<double>(1, 2) + rotation_estimated.at<double>(2, 1)) / s;
                    qz = 0.25 * s;
                }
            }
            
            geometry_msgs::PoseStamped pose_msg;
            pose_msg.header.stamp = image_msg->header.stamp;
            pose_msg.header.frame_id = frame_id_;
            pose_msg.pose.position.x = translation_estimated.at<double>(0);
            pose_msg.pose.position.y = translation_estimated.at<double>(1);
            pose_msg.pose.position.z = translation_estimated.at<double>(2);
            pose_msg.pose.orientation.w = qw;
            pose_msg.pose.orientation.x = qx;
            pose_msg.pose.orientation.y = qy;
            pose_msg.pose.orientation.z = qz;
            pose_pub_.publish(pose_msg);
            
            std::vector<cv::Point3f> axis_points;
            axis_points.push_back(cv::Point3f(0, 0, 0));
            axis_points.push_back(cv::Point3f(5, 0, 0));
            axis_points.push_back(cv::Point3f(0, 5, 0));
            axis_points.push_back(cv::Point3f(0, 0, 5));
            
            std::vector<cv::Point2f> projected_points;
            cv::projectPoints(axis_points, rvec, tvec, camera_matrix_, dist_coeffs_, projected_points);
            
            cv::line(frame_vis, projected_points[0], projected_points[1], cv::Scalar(0, 0, 255), 3);
            cv::line(frame_vis, projected_points[0], projected_points[2], cv::Scalar(0, 255, 0), 3);
            cv::line(frame_vis, projected_points[0], projected_points[3], cv::Scalar(255, 0, 0), 3);
            
            cv::drawMatches(frame, scene_keypoints, frame, scene_keypoints, good_matches, frame_vis);
        } else {
            ROS_DEBUG_THROTTLE(2.0, "Not enough inliers (%zu) for pose estimation (minimum: %d)", 
                             inliers.size(), min_inliers_kalman_);
        }
    } else {
        ROS_DEBUG_THROTTLE(2.0, "Not enough matches (%zu model, %zu scene) for PnP (minimum: 4)", 
                         matched_model_points.size(), matched_scene_points.size());
        // Draw detected keypoints even if we can't match
        cv::drawKeypoints(frame_vis, scene_keypoints, frame_vis, cv::Scalar(0, 255, 0));
    }
    
    sensor_msgs::ImagePtr vis_msg = cv_bridge::CvImage(image_msg->header, "bgr8", frame_vis).toImageMsg();
    vis_image_pub_.publish(vis_msg);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "opencv_pose_estimation");
    OpenCVPoseEstimation pose_estimator;
    ros::spin();
    return 0;
}

