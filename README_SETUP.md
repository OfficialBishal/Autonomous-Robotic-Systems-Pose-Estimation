# OpenCV Pose Estimation - Setup Guide

## Quick Setup

The OpenCV pose estimation is now configured to use the **mustard bottle** from your scene, which uses the existing YCB dataset mesh.

## Files Created

1. **Model file**: `config/opencv_model_mustard.yml` - Contains 3D points for mustard bottle
2. **Launch file**: `launch/opencv_pose_estimation.launch` - Configured to use mustard model
3. **Scripts**:
   - `scripts/generate_mustard_model.py` - Auto-generate model from YCB mesh
   - `scripts/generate_model_from_mesh.py` - Generate model from mesh + reference image

## Building

1. Enter Docker: `cd /home/local/csc752/csc752 && sh run.sh`
2. Inside Docker: `c` then `s`
3. Build: `cd hsr_robocanes_omniverse && catkin build final-project-OfficialBishal`

## Running

1. **Terminal 1** (inside Docker):
   ```bash
   python final-project-start.py
   ```

2. **Terminal 2** (inside Docker):
   ```bash
   s    # source workspace
   roslaunch final-project-OfficialBishal opencv_pose_estimation.launch
   ```

## Current Status

✅ **Model file exists** - Uses mustard bottle 3D points from YCB dataset  
⚠️ **Placeholder descriptors** - The descriptors are placeholders, so matching may not work perfectly  
✅ **Node will run** - It will process images and extract features  

## To Improve Tracking

The current model has placeholder descriptors. For real tracking:

1. **Take a reference image** of the mustard bottle in your scene
2. **Run the model generator** (inside Docker where OpenCV is available):
   ```bash
   python src/final-project-OfficialBishal/scripts/generate_model_from_mesh.py \
     --mesh src/usd/ycb/006_mustard_bottle/google_16k/nontextured.ply \
     --image path/to/your/reference_image.jpg \
     --output src/final-project-OfficialBishal/config/opencv_model_mustard.yml
   ```

Or use the auto-generator:
```bash
python src/final-project-OfficialBishal/scripts/generate_mustard_model.py
```

## Using Other Objects

To track other objects (cracker box, sugar box, etc.):

1. Find the mesh: `src/usd/ycb/XXX_object_name/google_16k/nontextured.ply`
2. Generate model: Use `generate_model_from_mesh.py` with that mesh
3. Update launch file: Change `model_file` parameter

## Topics

**Subscribed:**
- `/hsrb/head_rgbd_sensor/rgb/image_rect_color`
- `/hsrb/head_rgbd_sensor/rgb/camera_info`

**Published:**
- `~pose` - Estimated pose (when matches found)
- `~visualization_image` - Image with features/axes

