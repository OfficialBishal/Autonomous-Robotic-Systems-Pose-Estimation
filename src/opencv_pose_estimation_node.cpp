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
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/video/tracking.hpp>

// OpenCV PnP tutorial helper classes
#include "opencv_pnp/Model.h"
#include "opencv_pnp/RobustMatcher.h"
#include "opencv_pnp/PnPProblem.h"
#include "opencv_pnp/Utils.h"
#include "opencv_pnp/Mesh.h"

using namespace cv;
using namespace std;

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
    
    // OpenCV components
    Model model_;
    RobustMatcher rmatcher_;
    PnPProblem* pnp_detection_;
    PnPProblem* pnp_detection_est_;  // for Kalman filtered pose
    KalmanFilter kf_;
    
    // Camera parameters
    double camera_params_[4];  // fx, fy, cx, cy
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    
    // Feature detector/descriptor
    cv::Ptr<cv::Feature2D> detector_;
    cv::Ptr<cv::Feature2D> descriptor_;
    
    // Parameters
    std::string model_file_path_;
    std::string image_topic_;
    std::string camera_info_topic_;
    std::string frame_id_;
    int num_keypoints_;
    float ratio_test_;
    int ransac_iterations_;
    float ransac_reprojection_error_;
    double ransac_confidence_;
    int min_inliers_kalman_;
    int pnp_method_;
    std::string feature_name_;
    bool use_flann_;
    bool fast_match_;
    bool publish_visualization_;
    
    bool kalman_initialized_;
    double dt_;  // time step for Kalman filter
    
    // Helper functions
    void initKalmanFilter();
    void fillMeasurements(cv::Mat& measurements, const cv::Mat& translation, const cv::Mat& rotation);
    void updateKalmanFilter(cv::Mat& measurements, cv::Mat& translation_estimated, cv::Mat& rotation_estimated);
    cv::Mat rotationMatrixToQuaternion(const cv::Mat& R);
    
public:
    OpenCVPoseEstimation();
    ~OpenCVPoseEstimation();
    void imageCallback(const sensor_msgs::ImageConstPtr& image_msg, 
                      const sensor_msgs::CameraInfoConstPtr& info_msg);
};

OpenCVPoseEstimation::OpenCVPoseEstimation() 
    : nh_("~"), it_(nh_), 
      image_sub_(nh_, "", 1),
      info_sub_(nh_, "", 1),
      sync_(SyncPolicy(10), image_sub_, info_sub_),
      pnp_detection_(nullptr),
      pnp_detection_est_(nullptr),
      kalman_initialized_(false),
      dt_(0.033)  // ~30 FPS default
{
    // Load ROS parameters FIRST (before subscribing)
    nh_.param("model_file", model_file_path_, std::string(""));
    nh_.param("image_topic", image_topic_, std::string("/hsrb/head_rgbd_sensor/rgb/image_rect_color"));
    nh_.param("camera_info_topic", camera_info_topic_, std::string("/hsrb/head_rgbd_sensor/rgb/camera_info"));
    nh_.param("frame_id", frame_id_, std::string("head_rgbd_sensor_rgb_frame"));
    nh_.param("num_keypoints", num_keypoints_, 2000);
    nh_.param("ratio_test", ratio_test_, 0.7f);
    nh_.param("ransac_iterations", ransac_iterations_, 500);
    nh_.param("ransac_reprojection_error", ransac_reprojection_error_, 6.0f);
    nh_.param("ransac_confidence", ransac_confidence_, 0.99);
    nh_.param("min_inliers_kalman", min_inliers_kalman_, 30);
    nh_.param("pnp_method", pnp_method_, (int)cv::SOLVEPNP_ITERATIVE);
    nh_.param("feature", feature_name_, std::string("ORB"));
    nh_.param("use_flann", use_flann_, false);
    nh_.param("fast_match", fast_match_, true);
    nh_.param("publish_visualization", publish_visualization_, true);
    
    // Check if model file is specified
    if (model_file_path_.empty()) {
        ROS_ERROR("Model file path not specified! Please set the 'model_file' parameter.");
        return;
    }
    
    // Load 3D model
    ROS_INFO("Loading 3D model from: %s", model_file_path_.c_str());
    try {
        model_.load(model_file_path_);
        ROS_INFO("Loaded model with %d 3D points and %d descriptors", 
                 (int)model_.get_points3d().size(), model_.get_numDescriptors());
    } catch (const cv::Exception& e) {
        ROS_ERROR("Failed to load model file: %s", e.what());
        return;
    }
    
    if (model_.get_numDescriptors() == 0) {
        ROS_ERROR("Model file contains no descriptors!");
        return;
    }
    
    // Initialize feature detector and descriptor
    createFeatures(feature_name_, num_keypoints_, detector_, descriptor_);
    rmatcher_.setFeatureDetector(detector_);
    rmatcher_.setDescriptorExtractor(descriptor_);
    rmatcher_.setDescriptorMatcher(createMatcher(feature_name_, use_flann_));
    rmatcher_.setRatio(ratio_test_);
    
    // Set training image if available
    if (!model_.get_trainingImagePath().empty()) {
        cv::Mat training_img = cv::imread(model_.get_trainingImagePath());
        if (!training_img.empty()) {
            rmatcher_.setTrainingImage(training_img);
            ROS_INFO("Loaded training image: %s", model_.get_trainingImagePath().c_str());
        }
    }
    
    // Initialize Kalman filter
    initKalmanFilter();
    
    // Initialize camera parameters (will be updated from camera_info)
    camera_params_[0] = 500.0;  // fx (will be updated)
    camera_params_[1] = 500.0;  // fy (will be updated)
    camera_params_[2] = 320.0;  // cx (will be updated)
    camera_params_[3] = 240.0;  // cy (will be updated)
    
    // Initialize PnP solvers (will be properly initialized when we get camera info)
    pnp_detection_ = new PnPProblem(camera_params_);
    pnp_detection_est_ = new PnPProblem(camera_params_);
    
    // Setup subscribers with topic names from parameters
    ROS_INFO("Subscribing to image topic: %s", image_topic_.c_str());
    ROS_INFO("Subscribing to camera_info topic: %s", camera_info_topic_.c_str());
    
    image_sub_.subscribe(nh_, image_topic_, 1);
    info_sub_.subscribe(nh_, camera_info_topic_, 1);
    
    // Register callback
    sync_.registerCallback(boost::bind(&OpenCVPoseEstimation::imageCallback, this, _1, _2));
    
    ROS_INFO("Message filter synchronizer configured (queue size: 10, approximate time sync)");
    ROS_INFO("Note: Messages must arrive within ~0.1s of each other to be synchronized");
    
    // Setup publishers
    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("pose", 10);
    if (publish_visualization_) {
        vis_image_pub_ = it_.advertise("visualization_image", 1);
    }
    
    ROS_INFO("OpenCV Pose Estimation node initialized");
    ROS_INFO("Feature type: %s, Keypoints: %d, Ratio test: %.2f", 
             feature_name_.c_str(), num_keypoints_, ratio_test_);
    ROS_INFO("Node ready. Waiting for synchronized image and camera_info messages...");
    ROS_INFO("If no messages are received, check that both topics are publishing with similar timestamps");
}

OpenCVPoseEstimation::~OpenCVPoseEstimation() {
    if (pnp_detection_) delete pnp_detection_;
    if (pnp_detection_est_) delete pnp_detection_est_;
}

void OpenCVPoseEstimation::initKalmanFilter() {
    int nStates = 18;      // position + velocity + acceleration + rotation + angular velocity + angular acceleration
    int nMeasurements = 6; // x, y, z, roll, pitch, yaw
    int nInputs = 0;
    
    kf_.init(nStates, nMeasurements, nInputs, CV_64F);
    
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar::all(1e-5));
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1));
    
    // Position dynamics
    kf_.transitionMatrix.at<double>(0,3) = dt_;
    kf_.transitionMatrix.at<double>(1,4) = dt_;
    kf_.transitionMatrix.at<double>(2,5) = dt_;
    kf_.transitionMatrix.at<double>(3,6) = dt_;
    kf_.transitionMatrix.at<double>(4,7) = dt_;
    kf_.transitionMatrix.at<double>(5,8) = dt_;
    kf_.transitionMatrix.at<double>(0,6) = 0.5*pow(dt_,2);
    kf_.transitionMatrix.at<double>(1,7) = 0.5*pow(dt_,2);
    kf_.transitionMatrix.at<double>(2,8) = 0.5*pow(dt_,2);
    
    // Orientation dynamics
    kf_.transitionMatrix.at<double>(9,12) = dt_;
    kf_.transitionMatrix.at<double>(10,13) = dt_;
    kf_.transitionMatrix.at<double>(11,14) = dt_;
    kf_.transitionMatrix.at<double>(12,15) = dt_;
    kf_.transitionMatrix.at<double>(13,16) = dt_;
    kf_.transitionMatrix.at<double>(14,17) = dt_;
    kf_.transitionMatrix.at<double>(9,15) = 0.5*pow(dt_,2);
    kf_.transitionMatrix.at<double>(10,16) = 0.5*pow(dt_,2);
    kf_.transitionMatrix.at<double>(11,17) = 0.5*pow(dt_,2);
    
    // Measurement model
    kf_.measurementMatrix.at<double>(0,0) = 1;  // x
    kf_.measurementMatrix.at<double>(1,1) = 1;  // y
    kf_.measurementMatrix.at<double>(2,2) = 1;  // z
    kf_.measurementMatrix.at<double>(3,9) = 1;  // roll
    kf_.measurementMatrix.at<double>(4,10) = 1; // pitch
    kf_.measurementMatrix.at<double>(5,11) = 1; // yaw
    
    kalman_initialized_ = true;
}

void OpenCVPoseEstimation::fillMeasurements(cv::Mat& measurements, const cv::Mat& translation, const cv::Mat& rotation) {
    cv::Mat measured_eulers = rot2euler(rotation);
    measurements.at<double>(0) = translation.at<double>(0); // x
    measurements.at<double>(1) = translation.at<double>(1); // y
    measurements.at<double>(2) = translation.at<double>(2); // z
    measurements.at<double>(3) = measured_eulers.at<double>(0); // roll
    measurements.at<double>(4) = measured_eulers.at<double>(1); // pitch
    measurements.at<double>(5) = measured_eulers.at<double>(2); // yaw
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

cv::Mat OpenCVPoseEstimation::rotationMatrixToQuaternion(const cv::Mat& R) {
    // Convert rotation matrix to quaternion (ROS format: x, y, z, w)
    double trace = R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2);
    cv::Mat quat(4, 1, CV_64F);
    
    if (trace > 0) {
        double s = sqrt(trace + 1.0) * 2.0;
        quat.at<double>(3) = 0.25 * s;  // w
        quat.at<double>(0) = (R.at<double>(2,1) - R.at<double>(1,2)) / s;  // x
        quat.at<double>(1) = (R.at<double>(0,2) - R.at<double>(2,0)) / s;  // y
        quat.at<double>(2) = (R.at<double>(1,0) - R.at<double>(0,1)) / s;  // z
    } else if ((R.at<double>(0,0) > R.at<double>(1,1)) && (R.at<double>(0,0) > R.at<double>(2,2))) {
        double s = sqrt(1.0 + R.at<double>(0,0) - R.at<double>(1,1) - R.at<double>(2,2)) * 2.0;
        quat.at<double>(3) = (R.at<double>(2,1) - R.at<double>(1,2)) / s;  // w
        quat.at<double>(0) = 0.25 * s;  // x
        quat.at<double>(1) = (R.at<double>(0,1) + R.at<double>(1,0)) / s;  // y
        quat.at<double>(2) = (R.at<double>(0,2) + R.at<double>(2,0)) / s;  // z
    } else if (R.at<double>(1,1) > R.at<double>(2,2)) {
        double s = sqrt(1.0 + R.at<double>(1,1) - R.at<double>(0,0) - R.at<double>(2,2)) * 2.0;
        quat.at<double>(3) = (R.at<double>(0,2) - R.at<double>(2,0)) / s;  // w
        quat.at<double>(0) = (R.at<double>(0,1) + R.at<double>(1,0)) / s;  // x
        quat.at<double>(1) = 0.25 * s;  // y
        quat.at<double>(2) = (R.at<double>(1,2) + R.at<double>(2,1)) / s;  // z
    } else {
        double s = sqrt(1.0 + R.at<double>(2,2) - R.at<double>(0,0) - R.at<double>(1,1)) * 2.0;
        quat.at<double>(3) = (R.at<double>(1,0) - R.at<double>(0,1)) / s;  // w
        quat.at<double>(0) = (R.at<double>(0,2) + R.at<double>(2,0)) / s;  // x
        quat.at<double>(1) = (R.at<double>(1,2) + R.at<double>(2,1)) / s;  // y
        quat.at<double>(2) = 0.25 * s;  // z
    }
    
    return quat;
}

void OpenCVPoseEstimation::imageCallback(const sensor_msgs::ImageConstPtr& image_msg, 
                                        const sensor_msgs::CameraInfoConstPtr& info_msg) {
    static int callback_count = 0;
    callback_count++;
    ROS_INFO_THROTTLE(2.0, "Callback #%d: Received synchronized image (%dx%d) and camera_info", 
                     callback_count, image_msg->width, image_msg->height);
    
    // Convert ROS image to OpenCV
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
    cv::Mat frame = cv_ptr->image;
    cv::Mat frame_vis = frame.clone();
    
    ROS_DEBUG_THROTTLE(5.0, "Processing image: %dx%d", frame.cols, frame.rows);
    
    // Update camera parameters from camera_info
    camera_params_[0] = info_msg->K[0];  // fx
    camera_params_[1] = info_msg->K[4];  // fy
    camera_params_[2] = info_msg->K[2];  // cx
    camera_params_[3] = info_msg->K[5];  // cy
    
    // Update camera matrix
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
    
    // Update distortion coefficients
    dist_coeffs_ = cv::Mat(info_msg->D.size(), 1, CV_64F);
    for (size_t i = 0; i < info_msg->D.size(); i++) {
        dist_coeffs_.at<double>(i) = info_msg->D[i];
    }
    
    // Update PnP solvers with new camera parameters (if changed significantly)
    // For efficiency, we could check if params changed, but for simplicity we'll recreate
    if (pnp_detection_) delete pnp_detection_;
    if (pnp_detection_est_) delete pnp_detection_est_;
    pnp_detection_ = new PnPProblem(camera_params_);
    pnp_detection_est_ = new PnPProblem(camera_params_);
    
    // Get model data
    std::vector<cv::Point3f> list_points3d_model = model_.get_points3d();
    cv::Mat descriptors_model = model_.get_descriptors();
    std::vector<cv::KeyPoint> keypoints_model = model_.get_keypoints();
    
    // Step 1: Robust matching between model descriptors and scene descriptors
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::KeyPoint> keypoints_scene;
    
    if (fast_match_) {
        rmatcher_.fastRobustMatch(frame, good_matches, keypoints_scene, descriptors_model, keypoints_model);
    } else {
        rmatcher_.robustMatch(frame, good_matches, keypoints_scene, descriptors_model, keypoints_model);
    }
    
    ROS_DEBUG_THROTTLE(2.0, "Found %zu good matches out of model descriptors", good_matches.size());
    
    // Step 2: Extract 2D-3D correspondences
    std::vector<cv::Point3f> list_points3d_model_match;
    std::vector<cv::Point2f> list_points2d_scene_match;
    
    for (size_t match_index = 0; match_index < good_matches.size(); ++match_index) {
        cv::Point3f point3d_model = list_points3d_model[good_matches[match_index].trainIdx];
        cv::Point2f point2d_scene = keypoints_scene[good_matches[match_index].queryIdx].pt;
        list_points3d_model_match.push_back(point3d_model);
        list_points2d_scene_match.push_back(point2d_scene);
    }
    
    // Draw all matches (outliers)
    if (publish_visualization_) {
        draw2DPoints(frame_vis, list_points2d_scene_match, cv::Scalar(0, 0, 255));  // red
    }
    
    cv::Mat inliers_idx;
    bool good_measurement = false;
    
    // Step 3: Estimate pose using RANSAC
    if (good_matches.size() >= 4) {  // OpenCV requires at least 4 points
        ROS_DEBUG_THROTTLE(2.0, "Attempting PnP with %zu matches", good_matches.size());
        pnp_detection_->estimatePoseRANSAC(list_points3d_model_match, list_points2d_scene_match,
                                          pnp_method_, inliers_idx,
                                          ransac_iterations_, ransac_reprojection_error_, ransac_confidence_);
        
        ROS_DEBUG_THROTTLE(2.0, "PnP RANSAC found %d inliers", inliers_idx.rows);
        
        // Extract inliers
        std::vector<cv::Point2f> list_points2d_inliers;
        for (int inliers_index = 0; inliers_index < inliers_idx.rows; ++inliers_index) {
            int n = inliers_idx.at<int>(inliers_index);
            cv::Point2f point2d = list_points2d_scene_match[n];
            list_points2d_inliers.push_back(point2d);
        }
        
        // Draw inliers
        if (publish_visualization_) {
            draw2DPoints(frame_vis, list_points2d_inliers, cv::Scalar(255, 0, 0));  // blue
        }
        
        // Step 4: Kalman Filter
        if (inliers_idx.rows >= min_inliers_kalman_) {
            cv::Mat translation_measured = pnp_detection_->get_t_matrix();
            cv::Mat rotation_measured = pnp_detection_->get_R_matrix();
            
            cv::Mat measurements(6, 1, CV_64F);
            fillMeasurements(measurements, translation_measured, rotation_measured);
            good_measurement = true;
            
            // Update Kalman filter
            cv::Mat translation_estimated(3, 1, CV_64F);
            cv::Mat rotation_estimated(3, 3, CV_64F);
            updateKalmanFilter(measurements, translation_estimated, rotation_estimated);
            
            // Set estimated projection matrix
            pnp_detection_est_->set_P_matrix(rotation_estimated, translation_estimated);
            
            // Convert rotation matrix to quaternion
            cv::Mat quat = rotationMatrixToQuaternion(rotation_estimated);
            
            // Publish pose
            geometry_msgs::PoseStamped pose_msg;
            pose_msg.header.stamp = image_msg->header.stamp;
            pose_msg.header.frame_id = frame_id_;
            pose_msg.pose.position.x = translation_estimated.at<double>(0);
            pose_msg.pose.position.y = translation_estimated.at<double>(1);
            pose_msg.pose.position.z = translation_estimated.at<double>(2);
            pose_msg.pose.orientation.x = quat.at<double>(0);
            pose_msg.pose.orientation.y = quat.at<double>(1);
            pose_msg.pose.orientation.z = quat.at<double>(2);
            pose_msg.pose.orientation.w = quat.at<double>(3);
            pose_pub_.publish(pose_msg);
            
            ROS_INFO_THROTTLE(2.0, "Published pose: position=(%.3f, %.3f, %.3f), inliers=%d", 
                             translation_estimated.at<double>(0),
                             translation_estimated.at<double>(1),
                             translation_estimated.at<double>(2),
                             inliers_idx.rows);
            
            // Draw coordinate axes
            if (publish_visualization_) {
                float l = 5.0;
                std::vector<cv::Point2f> pose_points2d;
                pose_points2d.push_back(pnp_detection_est_->backproject3DPoint(cv::Point3f(0,0,0)));  // origin
                pose_points2d.push_back(pnp_detection_est_->backproject3DPoint(cv::Point3f(l,0,0)));  // x-axis
                pose_points2d.push_back(pnp_detection_est_->backproject3DPoint(cv::Point3f(0,l,0)));  // y-axis
                pose_points2d.push_back(pnp_detection_est_->backproject3DPoint(cv::Point3f(0,0,l)));  // z-axis
                draw3DCoordinateAxes(frame_vis, pose_points2d);
            }
        } else {
            ROS_WARN_THROTTLE(2.0, "Not enough inliers (%d) for Kalman update (minimum: %d)", 
                             inliers_idx.rows, min_inliers_kalman_);
        }
    } else {
        ROS_WARN_THROTTLE(2.0, "Not enough matches (%zu) for PnP (minimum: 4). Model descriptors may not match scene features.", good_matches.size());
    }
    
    // Publish visualization image
    if (publish_visualization_) {
        sensor_msgs::ImagePtr vis_msg = cv_bridge::CvImage(image_msg->header, "bgr8", frame_vis).toImageMsg();
        vis_image_pub_.publish(vis_msg);
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "opencv_pose_estimation");
    OpenCVPoseEstimation pose_estimator;
    ros::spin();
    return 0;
}
