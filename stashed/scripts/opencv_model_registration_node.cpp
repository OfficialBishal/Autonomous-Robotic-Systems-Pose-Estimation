#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

// OpenCV PnP tutorial helper classes
#include "opencv_pnp/Model.h"
#include "opencv_pnp/RobustMatcher.h"
#include "opencv_pnp/PnPProblem.h"
#include "opencv_pnp/Utils.h"
#include "opencv_pnp/Mesh.h"
#include "opencv_pnp/ModelRegistration.h"

using namespace cv;
using namespace std;

class OpenCVModelRegistration {
private:
    ros::NodeHandle nh_;
    ros::Subscriber image_sub_;
    ros::Subscriber info_sub_;
    
    Model model_;
    Mesh mesh_;
    ModelRegistration registration_;
    PnPProblem* pnp_registration_;
    RobustMatcher rmatcher_;
    
    cv::Mat reference_image_;
    cv::Mat camera_matrix_;
    double camera_params_[4];
    
    std::string mesh_file_path_;
    std::string output_model_path_;
    std::string image_topic_;
    std::string camera_info_topic_;
    int num_keypoints_;
    std::string feature_name_;
    
    bool image_received_;
    bool camera_info_received_;
    bool registration_complete_;
    
    // Points to register (8 corners of bounding box)
    const int num_registration_points_ = 8;
    const int corner_order_[8] = {0, 1, 2, 3, 4, 5, 6, 7}; // Front-top-right, front-top-left, etc.
    
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg);
    static void onMouse(int event, int x, int y, int flags, void* userdata);
    
public:
    OpenCVModelRegistration();
    ~OpenCVModelRegistration();
    void run();
};

// Static pointer for mouse callback
static OpenCVModelRegistration* g_registration_ptr = nullptr;

OpenCVModelRegistration::OpenCVModelRegistration() 
    : nh_("~"),
      pnp_registration_(nullptr),
      image_received_(false),
      camera_info_received_(false),
      registration_complete_(false)
{
    // Load parameters
    nh_.param("mesh_file", mesh_file_path_, std::string(""));
    nh_.param("output_model", output_model_path_, std::string("opencv_model_registered.yml"));
    nh_.param("image_topic", image_topic_, std::string("/hsrb/head_rgbd_sensor/rgb/image_rect_color"));
    nh_.param("camera_info_topic", camera_info_topic_, std::string("/hsrb/head_rgbd_sensor/rgb/camera_info"));
    nh_.param("num_keypoints", num_keypoints_, 2000);
    nh_.param("feature", feature_name_, std::string("ORB"));
    
    if (mesh_file_path_.empty()) {
        ROS_ERROR("Mesh file path not specified! Use 'mesh_file' parameter.");
        return;
    }
    
    // Load mesh
    ROS_INFO("Loading mesh from: %s", mesh_file_path_.c_str());
    mesh_.load(mesh_file_path_);
    ROS_INFO("Loaded mesh with %d vertices", mesh_.getNumVertices());
    
    // Initialize camera parameters (will be updated from camera_info)
    camera_params_[0] = 500.0;
    camera_params_[1] = 500.0;
    camera_params_[2] = 320.0;
    camera_params_[3] = 240.0;
    pnp_registration_ = new PnPProblem(camera_params_);
    
    // Initialize feature detector
    cv::Ptr<cv::Feature2D> detector, descriptor;
    createFeatures(feature_name_, num_keypoints_, detector, descriptor);
    rmatcher_.setFeatureDetector(detector);
    rmatcher_.setDescriptorExtractor(descriptor);
    
    // Setup registration
    registration_.setNumMax(num_registration_points_);
    
    // Setup subscribers
    image_sub_ = nh_.subscribe(image_topic_, 1, &OpenCVModelRegistration::imageCallback, this);
    info_sub_ = nh_.subscribe(camera_info_topic_, 1, &OpenCVModelRegistration::cameraInfoCallback, this);
    
    g_registration_ptr = this;
    
    ROS_INFO("OpenCV Model Registration node initialized");
    ROS_INFO("Waiting for image and camera_info...");
    ROS_INFO("Once received, a window will open for interactive registration");
}

OpenCVModelRegistration::~OpenCVModelRegistration() {
    if (pnp_registration_) delete pnp_registration_;
}

void OpenCVModelRegistration::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    if (image_received_) return; // Only use first image
    
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
    reference_image_ = cv_ptr->image.clone();
    image_received_ = true;
    ROS_INFO("Received reference image: %dx%d", reference_image_.cols, reference_image_.rows);
    
    if (camera_info_received_ && image_received_) {
        run();
    }
}

void OpenCVModelRegistration::cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
    if (camera_info_received_) return; // Only use first message
    
    // Update camera parameters
    camera_params_[0] = msg->K[0];  // fx
    camera_params_[1] = msg->K[4];  // fy
    camera_params_[2] = msg->K[2];  // cx
    camera_params_[3] = msg->K[5];  // cy
    
    // Update camera matrix
    camera_matrix_ = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 9; i++) {
        camera_matrix_.at<double>(i/3, i%3) = msg->K[i];
    }
    
    // Recreate PnP solver with correct parameters
    if (pnp_registration_) delete pnp_registration_;
    pnp_registration_ = new PnPProblem(camera_params_);
    
    camera_info_received_ = true;
    ROS_INFO("Received camera info: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f", 
             camera_params_[0], camera_params_[1], camera_params_[2], camera_params_[3]);
    
    if (camera_info_received_ && image_received_) {
        run();
    }
}

void OpenCVModelRegistration::onMouse(int event, int x, int y, int flags, void* userdata) {
    if (g_registration_ptr == nullptr) return;
    
    if (event == cv::EVENT_LBUTTONUP) {
        if (g_registration_ptr->registration_.is_registrable()) {
            int n_regist = g_registration_ptr->registration_.getNumRegist();
            int n_vertex = g_registration_ptr->corner_order_[n_regist];
            
            cv::Point2f point_2d((float)x, (float)y);
            cv::Point3f point_3d = g_registration_ptr->mesh_.getVertex(n_vertex);
            
            g_registration_ptr->registration_.registerPoint(point_2d, point_3d);
            
            ROS_INFO("Registered point %d/%d: 2D=(%.1f, %.1f), 3D=(%.3f, %.3f, %.3f)",
                     n_regist + 1, g_registration_ptr->num_registration_points_,
                     point_2d.x, point_2d.y, point_3d.x, point_3d.y, point_3d.z);
            
            if (g_registration_ptr->registration_.getNumRegist() == g_registration_ptr->num_registration_points_) {
                g_registration_ptr->registration_complete_ = true;
                ROS_INFO("Registration complete! Processing...");
            }
        }
    }
}

void OpenCVModelRegistration::run() {
    if (registration_complete_) return;
    
    ROS_INFO("Starting interactive registration...");
    ROS_INFO("Click on the 8 corners of the mustard bottle in the image");
    ROS_INFO("Order: Front-top-right, Front-top-left, Front-bottom-left, Front-bottom-right,");
    ROS_INFO("       Rear-top-right, Rear-top-left, Rear-bottom-left, Rear-bottom-right");
    
    cv::namedWindow("MODEL REGISTRATION", cv::WINDOW_KEEPRATIO);
    cv::setMouseCallback("MODEL REGISTRATION", onMouse, nullptr);
    
    cv::Mat img_vis;
    const cv::Scalar red(0, 0, 255);
    const cv::Scalar green(0, 255, 0);
    
    while (!registration_complete_ && cv::waitKey(30) < 0) {
        img_vis = reference_image_.clone();
        
        std::vector<cv::Point2f> list_points2d = registration_.get_points2d();
        std::vector<cv::Point3f> list_points3d = registration_.get_points3d();
        
        // Draw registered points
        if (!list_points2d.empty()) {
            drawPoints(img_vis, list_points2d, list_points3d, red);
        }
        
        // Draw current point to register
        if (!registration_complete_) {
            int n_regist = registration_.getNumRegist();
            int n_vertex = corner_order_[n_regist];
            cv::Point3f current_point3d = mesh_.getVertex(n_vertex);
            drawQuestion(img_vis, current_point3d, green);
            drawCounter(img_vis, registration_.getNumRegist(), num_registration_points_, red);
        }
        
        cv::imshow("MODEL REGISTRATION", img_vis);
    }
    
    if (!registration_complete_) {
        ROS_WARN("Registration cancelled or incomplete");
        return;
    }
    
    // Compute camera pose
    ROS_INFO("Computing camera pose from registered points...");
    std::vector<cv::Point2f> list_points2d = registration_.get_points2d();
    std::vector<cv::Point3f> list_points3d = registration_.get_points3d();
    
    bool is_correspondence = pnp_registration_->estimatePose(list_points3d, list_points2d, cv::SOLVEPNP_ITERATIVE);
    if (!is_correspondence) {
        ROS_ERROR("Failed to compute camera pose!");
        return;
    }
    
    ROS_INFO("Camera pose computed successfully");
    
    // Extract features from reference image
    ROS_INFO("Extracting features from reference image...");
    std::vector<cv::KeyPoint> keypoints_model;
    cv::Mat descriptors;
    
    rmatcher_.computeKeyPoints(reference_image_, keypoints_model);
    rmatcher_.computeDescriptors(reference_image_, keypoints_model, descriptors);
    
    ROS_INFO("Found %zu keypoints in reference image", keypoints_model.size());
    
    // Match features to 3D points
    ROS_INFO("Matching features to 3D mesh points...");
    for (size_t i = 0; i < keypoints_model.size(); ++i) {
        cv::Point2f point2d(keypoints_model[i].pt);
        cv::Point3f point3d;
        bool on_surface = pnp_registration_->backproject2DPoint(&mesh_, point2d, point3d);
        if (on_surface) {
            model_.add_correspondence(point2d, point3d);
            model_.add_descriptor(descriptors.row(i));
            model_.add_keypoint(keypoints_model[i]);
        } else {
            model_.add_outlier(point2d);
        }
    }
    
    ROS_INFO("Matched %zu features to 3D points", model_.get_points3d().size());
    
    // Save model
    model_.set_trainingImagePath(image_topic_); // Store topic name as reference
    model_.save(output_model_path_);
    
    ROS_INFO("Model saved to: %s", output_model_path_.c_str());
    ROS_INFO("Model contains %d 3D points with descriptors", model_.get_numDescriptors());
    
    // Show final result
    img_vis = reference_image_.clone();
    std::vector<cv::Point2f> list_points_in = model_.get_points2d_in();
    std::vector<cv::Point2f> list_points_out = model_.get_points2d_out();
    
    draw2DPoints(img_vis, list_points_in, green);
    draw2DPoints(img_vis, list_points_out, red);
    drawObjectMesh(img_vis, &mesh_, pnp_registration_, cv::Scalar(255, 0, 0));
    
    std::string text = "Model saved! Press any key to exit.";
    drawText(img_vis, text, green);
    cv::imshow("MODEL REGISTRATION", img_vis);
    
    ROS_INFO("Press any key in the window to exit...");
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    registration_complete_ = true;
    ros::shutdown();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "opencv_model_registration");
    OpenCVModelRegistration registration;
    ros::spin();
    return 0;
}




