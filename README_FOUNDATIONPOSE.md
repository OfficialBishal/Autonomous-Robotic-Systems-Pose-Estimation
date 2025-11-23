# FoundationPose Pose Estimation for ROS

This package integrates FoundationPose 6D pose estimation with ROS1 for use in the final project.

## Prerequisites

1. **FoundationPose Environment**: Make sure you have the `foundationpose` conda environment set up and working (see FoundationPose README for setup instructions).

2. **Object Mesh**: You need a 3D mesh file (`.obj` format) of the object you want to track. Place it in the `meshes/` directory or update the path in the launch file.

3. **Camera Topics**: The node subscribes to RGB, depth, and camera info topics. Default topics are for HSR robot in Isaac Sim:
   - RGB: `/hsrb/head_rgbd_sensor/rgb/image_rect_color`
   - Depth: `/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw`
   - Camera Info: `/hsrb/head_rgbd_sensor/rgb/camera_info`

## Usage

### 1. Basic Usage

Launch the FoundationPose node with default parameters:

```bash
roslaunch final-project-OfficialBishal foundationpose_pose_estimation.launch
```

### 2. With Custom Mesh

Specify a custom mesh file:

```bash
roslaunch final-project-OfficialBishal foundationpose_pose_estimation.launch \
  mesh_file:=$(rospack find final-project-OfficialBishal)/meshes/your_object.obj
```

### 3. With Custom Camera Topics

If your camera topics are different:

```bash
roslaunch final-project-OfficialBishal foundationpose_pose_estimation.launch \
  rgb_topic:=/your/camera/rgb/image \
  depth_topic:=/your/camera/depth/image \
  camera_info_topic:=/your/camera/rgb/camera_info
```

### 4. With Object Mask

If you have object segmentation available, you can provide a mask topic:

```bash
roslaunch final-project-OfficialBishal foundationpose_pose_estimation.launch \
  use_mask:=true \
  mask_topic:=/your/segmentation/mask/topic
```

## Parameters

- `mesh_file`: Path to the 3D mesh file (.obj format) of the object
- `rgb_topic`: ROS topic for RGB images
- `depth_topic`: ROS topic for depth images
- `camera_info_topic`: ROS topic for camera info (contains intrinsics)
- `frame_id`: Frame ID for the camera
- `object_frame_id`: Frame ID for the estimated object pose
- `est_refine_iter`: Number of refinement iterations for initial pose estimation (default: 5)
- `track_refine_iter`: Number of refinement iterations for tracking (default: 2)
- `debug`: Debug level (0=off, 1=basic, 2=detailed, 3=verbose)
- `use_mask`: Whether to use object mask for segmentation
- `mask_topic`: ROS topic for object mask (if use_mask is true)

## Published Topics

- `~pose` (`geometry_msgs/PoseStamped`): Estimated object pose in camera frame
- `~markers` (`visualization_msgs/MarkerArray`): Visualization markers for RViz

## TF Frames

The node publishes a TF transform from the camera frame to the object frame:
- Parent frame: Camera frame (specified by `frame_id` parameter)
- Child frame: Object frame (specified by `object_frame_id` parameter)

## Running in Conda Environment

Since FoundationPose requires the `foundationpose` conda environment, you need to run the node with that environment activated:

```bash
# Activate conda environment
conda activate foundationpose

# Source ROS workspace
source ~/hsr_robocanes_omniverse/devel/setup.bash

# Launch the node
roslaunch final-project-OfficialBishal foundationpose_pose_estimation.launch
```

Alternatively, you can create a wrapper script that activates the environment before running the node.

## Troubleshooting

### Import Errors

If you see import errors for FoundationPose modules:
- Make sure you're in the `foundationpose` conda environment
- Check that FoundationPose is installed at `~/hsr_robocanes_omniverse/FoundationPose`
- Verify that all FoundationPose dependencies are installed

### Mesh Not Found

- Check that the mesh file path is correct
- Ensure the mesh file exists and is readable
- Use absolute paths or `$(find package_name)` ROS path substitution

### No Pose Estimates

- Check that camera topics are publishing data: `rostopic echo /hsrb/head_rgbd_sensor/rgb/image_rect_color`
- Verify camera info is available: `rostopic echo /hsrb/head_rgbd_sensor/rgb/camera_info`
- Ensure depth images are in meters (32FC1 encoding)
- If using mask, verify mask topic is publishing

### Poor Pose Estimates

- Ensure good lighting and object visibility
- Use proper object segmentation mask if available
- Adjust `est_refine_iter` and `track_refine_iter` parameters
- Check that mesh file matches the actual object

## Integration with Final Project

To integrate this with your final project scene:

1. Place your object mesh in the `meshes/` directory
2. Update the launch file with the correct mesh path
3. Ensure camera topics match your Isaac Sim setup
4. Launch the node alongside your simulation

The estimated pose will be published on `~pose` topic and as a TF transform, which can be used by other nodes in your pipeline.

