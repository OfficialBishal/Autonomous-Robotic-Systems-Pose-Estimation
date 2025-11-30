# FoundationPose + SAM Integration - Complete Setup Guide

## Overview

This setup integrates **FoundationPose** (6D pose estimation) with **SAM** (Segment Anything Model) for accurate object segmentation, enabling precise pose estimation for robotic manipulation tasks.

### System Architecture

RGB images come from ROS, go to SAM segmentation node (in sam conda env), which generates object masks. The masks are published to a ROS topic, and FoundationPose node (in foundationpose conda env) uses those masks along with RGB-D data to estimate 6D pose.

### Key Components

1. **SAM (Segment Anything Model)**: Generates high-quality object masks
   - Repository: https://github.com/facebookresearch/segment-anything
   - Location: `~/hsr_robocanes_omniverse/segment-anything/`
   - Conda Environment: `sam`
   - Purpose: Segments mustard bottle (excludes table) for accurate pose estimation

2. **FoundationPose**: 6D pose estimation from RGB-D images
   - Location: `~/hsr_robocanes_omniverse/src/FoundationPose/`
   - Conda Environment: `foundationpose`
   - Purpose: Estimates object pose using RGB, depth, and mask

3. **ROS1 Integration**: Connects everything together
   - ROS1 Noetic installed system-wide in Docker
   - Both nodes run in separate conda environments
   - Communication via ROS topics

## Why This Architecture?

### Problem Solved

Without SAM (using depth-based mask):
- Mask includes table pixels, causing height error around 30cm
- Position error around 35cm, orientation error around 26 degrees
- Not accurate enough for precise picking

With SAM (proper segmentation):
- Mask excludes table, height error less than 2cm
- Position error less than 5cm, orientation error less than 10 degrees
- Good enough for precise top-down picking

### Environment Separation

1. **FoundationPose** (`foundationpose` conda env):
   - Specific PyTorch/CUDA versions
   - FoundationPose dependencies
   - Needs ROS integration

2. **SAM** (`sam` conda env):
   - Different PyTorch version requirements
   - SAM-specific dependencies
   - Can run independently

3. **ROS1** (system-wide):
   - Installed in Docker container
   - Shared by both environments via PYTHONPATH

## Installation

### Prerequisites

- Docker container with ROS1 Noetic
- Conda installed
- NVIDIA GPU with CUDA 12.8 (or compatible)
- At least 8GB VRAM (for ViT-H) or 4GB VRAM (for ViT-B)

### Step 1: FoundationPose Setup

FoundationPose should already be set up. Verify:

```bash
conda activate foundationpose
python -c "from estimater import FoundationPose; print('FoundationPose OK')"
```

If not set up, see FoundationPose documentation.

### Step 2: SAM Setup

**Option A: Automated Setup (Recommended)**

```bash
cd ~/hsr_robocanes_omniverse/src/final-project-OfficialBishal
./setup/setup_sam.sh
```

**Option B: Manual Setup**

```bash
# 1. Clone SAM repository
cd ~/hsr_robocanes_omniverse
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything

# 2. Create conda environment
conda create -n sam python=3.9 -y
conda activate sam

# 3. Install PyTorch (CUDA 12.1 compatible with 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# 5. Install dependencies
pip install rospkg catkin_pkg opencv-python pycocotools matplotlib numpy

# 6. Install YOLO (for object detection strategy)
pip install ultralytics

# 7. Download checkpoint (choose one)
mkdir -p checkpoints
# ViT-B (fastest, ~375MB, ~4GB VRAM)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b.pth
# OR ViT-L (balanced, ~1.2GB, ~6GB VRAM)
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O checkpoints/sam_vit_l.pth
# OR ViT-H (best, ~2.4GB, ~8GB VRAM)
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h.pth
```

**Note**: YOLO models (e.g., `yolov8n.pt`) will be automatically downloaded on first use if not already present.

### Step 3: Install YOLO (for Object Detection Strategy)

**Quick Check:**
```bash
conda activate sam
python -c "from ultralytics import YOLO; print('YOLO OK')" || echo "YOLO not installed"
```

**If YOLO is not installed, install it:**
```bash
conda activate sam
pip install ultralytics
```

### Step 3: Verify Installation

```bash
# Verify SAM
conda activate sam
python -c "from segment_anything import sam_model_registry; print('SAM OK')"

# Verify YOLO (for object detection strategy)
python -c "from ultralytics import YOLO; print('YOLO OK')" || echo "YOLO not installed (optional)"

# Verify FoundationPose
conda activate foundationpose
python -c "from estimater import FoundationPose; print('FoundationPose OK')"
```

## How It Works

### Wrapper Scripts

Both nodes use wrapper scripts that:
1. **Source ROS setup** - Makes ROS commands and Python packages available
2. **Source ROS workspace** - Makes `rospack find` work
3. **Activate conda environment** - Gets dependencies
4. **Set PYTHONPATH** - Adds ROS Python packages so conda Python can import them

**FoundationPose wrapper** (`scripts/run_foundationpose.sh`):
- Activates `foundationpose` conda environment
- Sets up libffi to avoid cv_bridge conflicts
- Adds FoundationPose to PYTHONPATH

**SAM wrapper** (`scripts/run_sam_segmentation.sh`):
- Activates `sam` conda environment
- Sets up ROS integration
- Simpler (no libffi issues)

### ROS Topics

**Input Topics:**
- `/hsrb/head_rgbd_sensor/rgb/image_rect_color` - RGB images
- `/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw` - Depth images
- `/hsrb/head_rgbd_sensor/rgb/camera_info` - Camera intrinsics

**Intermediate Topics:**
- `/segmentation/mustard_mask` - Object mask from SAM (mono8 Image)

**Output Topics:**
- `~pose` - Estimated pose (PoseStamped)
- `~markers` - Visualization markers (MarkerArray)
- `/mustard_bottle/ground_truth_pose` - Ground truth (subscribed for comparison)

**TF Frames:**
- `head_rgbd_sensor_rgb_frame` - Camera frame
- `object_pose` - Object pose frame (published by FoundationPose)

## Running the System

### Option 1: Combined Launch (Recommended)

Runs both SAM and FoundationPose together. All parameters are loaded from the config file:

```bash
# Inside Docker container
source ~/hsr_robocanes_omniverse/devel/setup.bash
roslaunch final-project-OfficialBishal foundationpose_with_sam.launch
```

All parameters are in `config/foundationpose_config.yaml`. Just edit the config file to change settings - no need to pass arguments. The default configuration uses the `detection` strategy with YOLO for automatic object detection.

This will:
- Start SAM segmentation node (generates masks)
- Start FoundationPose node (uses masks for pose estimation)
- Connect mask topic between nodes
- Use YOLO to automatically detect target object and segment it (if detection strategy is enabled)
- Only publish poses when quality is acceptable for picking (position error < 0.2m, orientation error < 45 degrees)

### Option 2: Separate Nodes

**Terminal 1: SAM Segmentation**
```bash
conda activate sam
source ~/hsr_robocanes_omniverse/devel/setup.bash
rosrun final-project-OfficialBishal sam_segmentation_node.py
```

**Terminal 2: FoundationPose**
```bash
conda activate foundationpose
source ~/hsr_robocanes_omniverse/devel/setup.bash
roslaunch final-project-OfficialBishal foundationpose_pose_estimation.launch use_mask:=true mask_topic:=/segmentation/mustard_mask
```

### Option 3: FoundationPose Only (No SAM)

If you want to test without SAM (uses depth-based mask):

```bash
conda activate foundationpose
source ~/hsr_robocanes_omniverse/devel/setup.bash
roslaunch final-project-OfficialBishal foundationpose_pose_estimation.launch use_mask:=false
```

**Note**: This will have higher errors (~30cm height error) due to table pixels in mask.

## Configuration

### Configuration File System

All parameters are organized in a single configuration file: `config/foundationpose_config.yaml`. The launch file loads this config automatically - no arguments needed.

The config file has sections for:
- Mesh configuration
- Camera topics
- FoundationPose parameters
- SAM parameters
- Object detection (YOLO) parameters
- Orientation correction parameters
- Pose stability parameters
- Pose publishing quality thresholds

Paths in the config file can be relative to workspace root (e.g., `src/final-project-OfficialBishal/meshes/mustard.obj`), absolute paths, or paths starting with `~` for home directory. The Python code automatically resolves relative paths.

To change settings, just edit the config file and restart the launch file.

### SAM Segmentation Strategy

The SAM node supports different segmentation strategies (configured in `config/foundationpose_config.yaml`):

1. `center_point`: Uses center of image as prompt. Good if object is centered, but may fail if table is centered instead.

2. `point`: Uses specific point coordinates. Set `sam/prompt_point_x` and `sam/prompt_point_y` (0.0-1.0 for fraction, or pixel coordinates). Good when you know object location.

3. `box`: Uses bounding box. Set `sam/box_x_min`, `sam/box_y_min`, `sam/box_x_max`, `sam/box_y_max` (0.0-1.0 fractions). Good when you have object detector.

4. `automatic`: Generates all masks, filters by size. Set `sam/min_mask_area` and `sam/max_mask_area`. Slower but fully automatic. May need better filtering to exclude table.

5. `detection` (recommended): Uses YOLO object detection to find bounding box, then segments. Automatically detects target object (e.g., "bottle", "mustard") and uses detected bounding box as SAM prompt. Most robust strategy. Requires YOLO to be installed in the sam conda environment. Configuration parameters are in the `object_detection/` section of the config file.

### Model Selection

SAM Model options (configured in config file):
- `vit_b`: Fastest, least accurate (around 4GB VRAM) - good for testing
- `vit_l`: Balanced (around 6GB VRAM)
- `vit_h`: Most accurate (around 8GB VRAM) - good for production

FoundationPose parameters (configured in config file):
- `foundationpose/est_refine_iter`: Refinement iterations for initial registration (default: 5)
- `foundationpose/track_refine_iter`: Refinement iterations for tracking (default: 2)
- `foundationpose/debug`: Debug level (0=off, 1=basic, 2=detailed, 3=verbose)

### Pose Stability and Quality Control

Pose stability checking replaces the fixed delay. The system automatically detects when FoundationPose has converged by tracking recent pose estimates and analyzing variance. It only triggers orientation correction testing when the pose is stable. Configuration is in the `pose_stability/` section of the config file.

Pose publishing quality thresholds ensure only accurate poses are published for picking. If position error is less than 0.2m and orientation error is less than 45 degrees, the pose is published. Otherwise, it's not published (but ground truth comparison still runs for debugging). Configuration is in the `publish_quality/` section of the config file.

The config file has sections for mesh, camera, foundationpose, mask, depth_mask, orientation_correction, pose_stability, publish_quality, sam, and object_detection. See the actual file for the full structure.

Paths can be relative to workspace root (starting with `src/`), absolute, or use `~` for home directory. The code automatically resolves relative paths.

To modify configuration, just edit the config file and restart the launch file. The code accesses nested parameters like `rospy.get_param('~camera/rgb_topic')` and has fallbacks for backward compatibility.

## File Locations

### FoundationPose
- Code: `~/hsr_robocanes_omniverse/src/FoundationPose/`
- Conda Environment: `~/.conda/envs/foundationpose/`
- ROS Node: `~/hsr_robocanes_omniverse/src/final-project-OfficialBishal/scripts/foundationpose_pose_estimation_node.py`
- Wrapper Script: `~/hsr_robocanes_omniverse/src/final-project-OfficialBishal/scripts/run_foundationpose.sh`

### SAM
- Repository: `~/hsr_robocanes_omniverse/segment-anything/`
- Checkpoints: `~/hsr_robocanes_omniverse/segment-anything/checkpoints/`
- Conda Environment: `~/.conda/envs/sam/`
- ROS Node: `~/hsr_robocanes_omniverse/src/final-project-OfficialBishal/scripts/sam_segmentation_node.py`
- Wrapper Script: `~/hsr_robocanes_omniverse/src/final-project-OfficialBishal/scripts/run_sam_segmentation.sh`

### ROS
- ROS Python packages: `/opt/ros/noetic/lib/python3/dist-packages/`
- Launch files: `~/hsr_robocanes_omniverse/src/final-project-OfficialBishal/launch/`
- Config files: `~/hsr_robocanes_omniverse/src/final-project-OfficialBishal/config/`

## Current Implementation Status

### Completed

1. FoundationPose ROS Node
   - Subscribes to RGB, depth, camera info
   - Supports mask input (from SAM)
   - Publishes pose estimates and visualizations
   - Compares with ground truth from Isaac Sim

2. SAM ROS Node
   - Subscribes to RGB images
   - Generates object masks using SAM
   - Publishes masks for FoundationPose
   - Supports multiple segmentation strategies including YOLO-based detection

3. Integration
   - Combined launch file
   - Separate conda environments
   - ROS topic communication
   - Wrapper scripts for environment management

4. Recent Improvements
   - Configuration file system - all parameters organized in config file
   - Pose stability checking - automatically detects convergence (replaces fixed delay)
   - Pose quality thresholds - only publishes poses suitable for picking
   - Path resolution - automatic resolution of relative paths in config file
   - YOLO integration - automatic object detection for robust segmentation

### Known Issues / Limitations

1. SAM Segmentation Strategy
   - Default `center_point` may segment table if object not centered
   - Fixed: Use `detection` strategy with YOLO for automatic object detection
   - Alternative: Use depth-assisted point selection

2. Performance
   - SAM inference can be slow (around 100-200ms per frame)
   - Consider using smaller model (ViT-B) for real-time
   - Or run SAM at lower frequency than FoundationPose tracking

## Expected Performance

### With SAM Segmentation (Proper Mask)

- Position Error: less than 5 cm (from around 35 cm without SAM)
- Orientation Error: less than 10 degrees (from around 26 degrees)
- Height Error: less than 2 cm (from around 30 cm) - this is critical for picking
- Suitable for: Precise top-down picking

### Without SAM (Depth-Based Mask)

- Position Error: around 35 cm
- Orientation Error: around 26 degrees
- Height Error: around 30 cm - too high for picking
- Suitable for: Approximate localization only

## Troubleshooting

### "No module named 'segment_anything'"
- Verify SAM is installed: `pip list | grep segment-anything`
- Check conda environment: `conda activate sam`
- Reinstall: `pip install git+https://github.com/facebookresearch/segment-anything.git`

### "No module named 'rospy'"
- Make sure ROS is sourced: `source /opt/ros/noetic/setup.bash`
- Check PYTHONPATH includes: `/opt/ros/noetic/lib/python3/dist-packages`
- Install ROS dependencies: `pip install rospkg catkin_pkg`

### "No module named 'estimater'"
- Check FoundationPose path: `~/hsr_robocanes_omniverse/src/FoundationPose/`
- Verify PYTHONPATH includes this path
- Check FoundationPose is properly set up

### "SAM checkpoint not found"
- Verify checkpoint path in config file: `config/foundationpose_config.yaml`
- Check `sam/sam_checkpoint_path` parameter
- Paths can be relative to workspace root (e.g., `src/final-project-OfficialBishal/sam-checkpoints/sam_vit_h.pth`)
- Check checkpoint exists: `ls ~/hsr_robocanes_omniverse/src/final-project-OfficialBishal/sam-checkpoints/`
- Download checkpoint (see Step 2 above)

### "Mesh file not found"
- Verify mesh path in config file: `config/foundationpose_config.yaml`
- Check `mesh_file` parameter
- Paths can be relative to workspace root (e.g., `src/final-project-OfficialBishal/meshes/mustard.obj`)
- Python code automatically resolves paths relative to workspace root

### "CUDA out of memory"
- Use smaller SAM model (ViT-B instead of ViT-H)
- Reduce image resolution (if possible)
- Close other GPU applications

### Mask is empty or wrong object
- Try different segmentation strategy
- Adjust point/box coordinates
- Check image topic is publishing: `rostopic echo /hsrb/head_rgbd_sensor/rgb/image_rect_color -n 1`
- Consider using depth-assisted point selection

### Transform errors (map to camera frame)
- Check TF tree: `rosrun tf view_frames`
- Verify frames exist: `rostopic echo /tf`
- Multi-hop transform is implemented, but may need intermediate frames

## Verification

### Test SAM Node Alone

```bash
conda activate sam
source ~/hsr_robocanes_omniverse/devel/setup.bash
rosrun final-project-OfficialBishal sam_segmentation_node.py

# In another terminal, check mask topic
rostopic echo /segmentation/mustard_mask -n 1
```

### Test FoundationPose Node Alone

```bash
conda activate foundationpose
source ~/hsr_robocanes_omniverse/devel/setup.bash
roslaunch final-project-OfficialBishal foundationpose_pose_estimation.launch use_mask:=false
```

### Test Combined System

```bash
source ~/hsr_robocanes_omniverse/devel/setup.bash
roslaunch final-project-OfficialBishal foundationpose_with_sam.launch
```

All parameters are loaded from `config/foundationpose_config.yaml`. Edit the config file to change settings.

### Verify Setup

```bash
# Test SAM
conda activate sam
export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH"
python -c "from segment_anything import sam_model_registry; print('SAM OK')"

# Test FoundationPose
conda activate foundationpose
export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:$HOME/hsr_robocanes_omniverse/src/FoundationPose:$PYTHONPATH"
python -c "import rospy; from estimater import FoundationPose; print('FoundationPose OK')"
```

## Future Improvements

The object detection strategy using YOLO has been implemented. The detection strategy uses YOLO to automatically detect target objects, extracts bounding boxes, and uses them as SAM prompts for accurate segmentation. This is more robust than the center_point strategy and works even when the object is not centered.

To use it, edit `config/foundationpose_config.yaml` and set `sam/segmentation_strategy` to `"detection"`.

Possible future enhancements:
- Depth-assisted point selection - use depth to find object automatically
- Better automatic filtering - generate all masks and filter by depth/size/shape
- Performance optimization - use smaller SAM model or run at lower frequency
- Integration with Isaac Sim segmentation if available

## Documentation References

- Main README: `README.md` - Quick start guide
- FoundationPose Setup: This file - Complete setup and usage guide
- SAM Setup: `readmes/README_SAM_SETUP.md` - SAM installation guide
- Pose Estimation Improvements: `docs/POSE_ESTIMATION_IMPROVEMENTS.md` - Technical details

## Benefits of This Approach

- No environment conflicts - ROS, SAM, and FoundationPose stay separate
- Uses existing infrastructure - no need to reinstall ROS
- Maintainable - clear separation of concerns
- Portable - works in any Docker container with ROS + conda
- Accurate - SAM masks enable precise pose estimation
- Flexible - can run with or without SAM

## Recent Updates

All parameters have been moved to a config file (`config/foundationpose_config.yaml`). The launch file just loads the config - no arguments needed. Parameters are organized logically by component.

Pose quality control has been added - the system only publishes poses when position error is less than 0.2m and orientation error is less than 45 degrees. This prevents picking failures by ensuring only accurate poses are published.

Pose stability checking replaces the fixed 10-second delay. The system now automatically detects when the pose has converged and triggers correction testing at the right time.

Table orientation has been fixed - the table is oriented with edges aligned to grid axes and set as kinematic to prevent rotation. This keeps the mustard bottle stable on the table.
