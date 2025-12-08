# How to Create OpenCV Model File for Pose Estimation

This guide explains how to create a proper OpenCV model file with real ORB descriptors from a reference image.

## Overview

The OpenCV pose estimation requires a model file containing:
- **3D points**: Coordinates of points on the object surface (in meters)
- **Descriptors**: ORB feature descriptors extracted from a reference image
- **Keypoints**: 2D locations of features in the reference image

## Method 1: Using ROS Registration Node (Easiest)

I've created a ROS node that makes this process easier.

### Step 1: Build the Registration Node

```bash
cd ~/hsr_robocanes_omniverse
catkin_make
```

### Step 2: Prepare Your Scene

1. Place the mustard bottle in your scene (Isaac Sim or real robot)
2. Position the camera so the bottle is clearly visible
3. Make sure the bottle fills a reasonable portion of the image

### Step 3: Run the Registration Node

```bash
source devel/setup.bash
roslaunch final-project-OfficialBishal opencv_model_registration.launch
```

### Step 4: Interactive Registration

1. A window will open showing the current camera image
2. The tool will prompt you to click on **8 corner points** of the mustard bottle
3. Click on the corners in this order:
   - **Front-top-right** corner
   - **Front-top-left** corner
   - **Front-bottom-left** corner
   - **Front-bottom-right** corner
   - **Rear-top-right** corner
   - **Rear-top-left** corner
   - **Rear-bottom-left** corner
   - **Rear-bottom-right** corner

4. After clicking all 8 points:
   - The tool computes the camera pose
   - Extracts ORB features from the image
   - Matches features to 3D mesh points
   - Saves the model file

5. The output file will be saved to:
   ```
   config/opencv_model_mustard_registered.yml
   ```

### Step 5: Use the New Model File

Update your launch file to use the new model:
```xml
<param name="model_file" value="$(find final-project-OfficialBishal)/config/opencv_model_mustard_registered.yml" />
```

## Method 2: Using OpenCV Tutorial Tool (Manual)

If you prefer to use the original OpenCV tutorial tool:

### Step 1: Build OpenCV Tutorial

```bash
cd /home/local/csc752/csc752/hsr_robocanes_omniverse/opencv
mkdir -p build
cd build
cmake ..
make example_tutorial_pnp_registration
```

### Step 2: Save a Reference Image

1. Capture an image of the mustard bottle from your camera
2. Save it as a JPG or PNG file
3. Note the full path

### Step 3: Get Camera Parameters

Get camera intrinsics from ROS:
```bash
rostopic echo /hsrb/head_rgbd_sensor/rgb/camera_info -n 1
```

Note the K matrix values:
- K[0] = fx (focal length x)
- K[4] = fy (focal length y)
- K[2] = cx (principal point x)
- K[5] = cy (principal point y)

### Step 4: Create a Simple PLY File (if needed)

If you don't have a PLY file, you can create a simple bounding box PLY file with 8 corners.

The mustard bottle mesh is at:
```
src/usd/ycb/006_mustard_bottle/google_16k/nontextured.ply
```

### Step 5: Run Registration Tool

```bash
cd /home/local/csc752/csc752/hsr_robocanes_omniverse/opencv/build/bin
./example_tutorial_pnp_registration \
  --image /path/to/your/reference_image.jpg \
  --mesh /home/local/csc752/csc752/hsr_robocanes_omniverse/src/usd/ycb/006_mustard_bottle/google_16k/nontextured.ply \
  --model /home/local/csc752/csc752/hsr_robocanes_omniverse/src/final-project-OfficialBishal/config/opencv_model_mustard_registered.yml \
  --keypoints 2000 \
  --feature ORB
```

**Note**: The OpenCV tutorial tool uses hardcoded camera parameters. You may need to modify `main_registration.cpp` to use your camera's intrinsics, or the ROS node method is easier.

## Method 3: Quick Test with Existing Image

If you have an image saved from ROS:

```bash
# Save an image from ROS topic
rosrun image_view image_saver image:=/hsrb/head_rgbd_sensor/rgb/image_rect_color _filename_format:=mustard_reference.jpg

# Then use Method 1 or 2 with this image
```

## Troubleshooting

### "Not enough matches" warnings

This means the placeholder descriptors in the current model file don't match real scene features. You **must** create a model file with real descriptors from a reference image.

### Mesh file not found

Check the mesh path:
```bash
ls -la ~/hsr_robocanes_omniverse/src/usd/ycb/006_mustard_bottle/google_16k/nontextured.ply
```

### Camera parameters mismatch

The registration tool needs accurate camera intrinsics. Use the values from `camera_info` topic.

### Registration window doesn't open

Make sure you have X11 forwarding enabled if running in Docker:
```bash
xhost +local:root
```

## What the Model File Contains

After registration, the YAML file will have:
- **points_3d**: Hundreds or thousands of 3D points on the object surface
- **descriptors**: Real ORB descriptors (32 bytes each) extracted from the reference image
- **keypoints**: 2D locations of features in the reference image
- **training_image_path**: Path to the reference image (for reference)

This is much better than the placeholder file with only 8 points!

## Next Steps

After creating the model file:
1. Update the launch file to use the new model file
2. Restart the pose estimation node
3. You should see many more matches and successful pose estimates




