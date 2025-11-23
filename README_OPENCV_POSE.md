# OpenCV Real-Time Pose Estimation

## Overview

This implementation is based on the [OpenCV Real-Time Pose Estimation Tutorial](https://docs.opencv.org/4.x/dc/d2c/tutorial_real_time_pose.html). We've adapted the algorithm from the tutorial into a ROS node.

## What We're Using vs. What We Built

### ✅ Using OpenCV's Algorithm (from tutorial):
- **ORB feature detection** - Detects distinctive points in images
- **FLANN-based matching** - Matches features between model and scene
- **PnP + RANSAC** - Solves Perspective-n-Point problem to estimate pose
- **Kalman Filter** - Smooths pose estimates over time

### 🔧 What We Adapted:
- **ROS Integration** - Wrapped the algorithm in a ROS node
- **ROS Topics** - Subscribes to camera images, publishes pose messages
- **ROS Parameters** - Made it configurable via launch files
- **Message Types** - Uses ROS sensor_msgs and geometry_msgs

## Why Do We Need a 3D Model?

This is a **feature-based approach**, not a learning-based one. Here's the difference:

### Feature-Based (What We're Using):
- **Requires**: A known 3D model of your object (points + descriptors)
- **How it works**: 
  1. Match features from current image to features in the model
  2. Use matched 2D-3D correspondences to solve PnP
  3. Estimate pose from the correspondences
- **Pros**: Fast, doesn't need training data, works with any textured object
- **Cons**: Requires a 3D model of the object

### Learning-Based (Alternative, like DOPE):
- **Requires**: A trained neural network (trained on many images)
- **How it works**: 
  1. Neural network detects object and estimates pose directly
  2. No explicit feature matching needed
- **Pros**: Can detect objects without a pre-defined model
- **Cons**: Needs training data, may be slower, requires GPU for best performance

## The Algorithm Flow

```
1. Load 3D Model (at startup)
   └─> Read YAML file with 3D points and ORB descriptors

2. For Each Camera Frame:
   ├─> Extract ORB features from image
   ├─> Match features with model descriptors (FLANN)
   ├─> Get 2D-3D correspondences
   ├─> Solve PnP with RANSAC → Get pose estimate
   ├─> Apply Kalman Filter → Smooth pose
   └─> Publish pose message
```

## Creating the 3D Model File

The model file (`opencv_model.yml`) contains:
- **3D coordinates** (x, y, z) of feature points in object's frame
- **ORB descriptors** for each point (for matching)

### Option 1: Use OpenCV's Model Registration Tool (Best)
The OpenCV tutorial includes a model registration application:
- Location: `opencv/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/`
- Takes: An image of your object + 3D mesh file
- Outputs: YAML file with 3D points and descriptors

### Option 2: Use Our Helper Script
We provide a helper script to get started:
```bash
python scripts/generate_model_from_image.py your_object_image.jpg -o config/opencv_model.yml
```
**Note**: This creates a template - you still need to add the 3D coordinates manually!

### Option 3: Manual Creation
1. Take a reference image of your object
2. Extract ORB features
3. For each feature, determine its 3D position
4. Create YAML file with points and descriptors

## Usage

1. **Create/obtain your 3D model file** (see above)

2. **Build the package**:
   ```bash
   cd /home/local/csc752/csc752/hsr_robocanes_omniverse
   catkin build final-project-OfficialBishal
   source devel/setup.bash
   ```

3. **Start the simulation** (in one terminal):
   ```bash
   python final-project-start.py
   ```

4. **Launch pose estimation** (in another terminal):
   ```bash
   roslaunch final-project-OfficialBishal opencv_pose_estimation.launch
   ```

## Topics

### Subscribed:
- `/hsrb/head_rgbd_sensor/rgb/image_rect_color` - Camera images
- `/hsrb/head_rgbd_sensor/rgb/camera_info` - Camera calibration info

### Published:
- `~pose` - Estimated pose as `geometry_msgs/PoseStamped`
- `~visualization_image` - Image with coordinate axes overlay

## Parameters

Configure via launch file:
- `model_file` - Path to YAML model file
- `num_keypoints` - Number of ORB keypoints to detect (default: 2000)
- `ratio_test` - Ratio test threshold for matching (default: 0.7)
- `ransac_iterations` - RANSAC iterations (default: 500)
- `ransac_reprojection_error` - Max reprojection error (default: 2.0)
- `ransac_confidence` - RANSAC confidence (default: 0.95)
- `min_inliers_kalman` - Min inliers for Kalman update (default: 30)

## References

- [OpenCV Tutorial: Real Time pose estimation](https://docs.opencv.org/4.x/dc/d2c/tutorial_real_time_pose.html)
- [OpenCV PnP Documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

