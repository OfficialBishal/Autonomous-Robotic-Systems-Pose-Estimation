# Creating OpenCV Model File Using Registration Tool

This guide explains how to create a proper OpenCV model file with real ORB descriptors from a reference image.

## Method 1: Using OpenCV Tutorial Registration Tool (Recommended)

### Step 1: Build the Registration Tool

The OpenCV tutorial includes a registration tool. You need to build it:

```bash
cd /home/local/csc752/csc752/hsr_robocanes_omniverse/opencv
mkdir -p build
cd build
cmake ..
make example_tutorial_pnp_registration
```

### Step 2: Prepare Files

You need:
1. **Reference image**: A clear image of the mustard bottle (save as JPG/PNG)
2. **PLY mesh file**: The 3D mesh of the mustard bottle
3. **Camera parameters**: Camera intrinsics (fx, fy, cx, cy)

### Step 3: Run Registration Tool

```bash
cd /home/local/csc752/csc752/hsr_robocanes_omniverse/opencv/build/bin
./example_tutorial_pnp_registration \
  --image /path/to/reference_image.jpg \
  --mesh /path/to/mustard_bottle.ply \
  --model /path/to/output_model.yml \
  --keypoints 2000 \
  --feature ORB
```

### Step 4: Interactive Registration

1. A window will open showing your reference image
2. The tool will prompt you to click on 8 corner points of the object
3. Click on the 3D mesh corners in the image (in order: front-top-right, front-top-left, etc.)
4. After clicking all 8 points, the tool will:
   - Compute the camera pose
   - Extract ORB features from the image
   - Project 3D mesh points to 2D
   - Match features to 3D points
   - Save the model file

### Step 5: Verify Model File

The output YAML file should contain:
- `points_3d`: 3D coordinates
- `descriptors`: ORB descriptors (real features from image)
- `keypoints`: 2D keypoint locations
- `training_image_path`: Path to reference image

## Method 2: Using ROS Node (Alternative)

If you want to create a ROS node that does this, you can create a registration node that:
1. Subscribes to an image topic
2. Loads a PLY mesh
3. Allows interactive clicking (or uses automatic feature detection)
4. Saves the model file

## Finding the Mustard Bottle Mesh

The mustard bottle PLY file should be at:
```
src/usd/ycb/006_mustard_bottle/google_16k/nontextured.ply
```

Or check:
```bash
find ~/hsr_robocanes_omniverse -name "*mustard*.ply"
```

## Camera Parameters

For the HSR robot camera, you can get camera parameters from:
```bash
rostopic echo /hsrb/head_rgbd_sensor/rgb/camera_info
```

The camera matrix K contains:
- K[0] = fx
- K[4] = fy  
- K[2] = cx
- K[5] = cy

## Quick Test

To test if the registration tool works, you can use the example data:
```bash
./example_tutorial_pnp_registration \
  --image samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/Data/resized_IMG_3875.JPG \
  --mesh samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/Data/box.ply \
  --model test_model.yml
```

## Notes

- The reference image should show the object clearly
- Good lighting helps feature detection
- The object should fill a reasonable portion of the image
- Click the 8 corners in the correct order (as shown in the tool)
- More keypoints (2000+) give better matching but slower processing




