# Final Project - Object Pose Estimation

This package provides 6D pose estimation for robotic object manipulation using FoundationPose, SAM (Segment Anything Model), and Grounded SAM for segmentation.

## Overview

This system integrates multiple state-of-the-art components for accurate object pose estimation:
- **FoundationPose**: 6D pose estimation from RGB-D images
- **SAM (Segment Anything Model)**: High-quality object segmentation
- **Grounded SAM**: Open-vocabulary object detection and segmentation (alternative to YOLO + SAM)
- **YOLO**: Object detection for automatic segmentation

## Quick Start

### 1. Setup

```bash
# Navigate to project directory
cd ~/hsr_robocanes_omniverse/src/final-project-OfficialBishal

# Option A: Setup SAM with YOLO (recommended for COCO dataset objects)
./setup/setup_sam.sh

# Option B: Setup Grounded SAM (recommended for open-vocabulary detection)
./setup/setup_grounded_sam.sh

# Verify FoundationPose is set up
conda activate foundationpose
python -c "from estimater import FoundationPose; print('OK')"
```

### 2. Run the System

#### Option A: Using SAM with YOLO (Default)

```bash
# Launch pose estimation system
source ~/hsr_robocanes_omniverse/devel/setup.bash
roslaunch final-project-OfficialBishal foundationpose_with_sam.launch
```

#### Option B: Using Grounded SAM (Open-Vocabulary)

```bash
# Launch pose estimation system with Grounded SAM
source ~/hsr_robocanes_omniverse/devel/setup.bash
roslaunch final-project-OfficialBishal foundationpose_with_grounded_sam.launch
```

#### Option C: FoundationPose Only (Manual Mask)

```bash
# Launch FoundationPose only (requires external mask topic)
roslaunch final-project-OfficialBishal foundationpose_pose_estimation.launch
```

**Configuration**: All parameters are in `config/foundationpose_config.yaml`. No need to pass launch arguments.

## Main Components

### ROS Nodes

#### Segmentation Nodes
- **`sam_segmentation_node.py`** - SAM segmentation with YOLO object detection
  - Uses YOLO to detect objects, then SAM to generate masks
  - Supports multiple segmentation strategies (detection, center_point, point, box, automatic)
  - Runs in `sam` conda environment

- **`grounded_sam_segmentation_node.py`** - Grounded SAM segmentation
  - Uses Grounding DINO for open-vocabulary object detection
  - Better for objects not in COCO dataset (e.g., "cracker box")
  - Text-based prompts (e.g., "cracker box", "bottle")
  - Runs in `grounded_sam` conda environment

#### Pose Estimation Node
- **`foundationpose_pose_estimation_node.py`** - FoundationPose 6D pose estimation
  - Subscribes to RGB-D images and segmentation masks
  - Estimates 6D pose (position + orientation)
  - Publishes poses with quality checking
  - Runs in `foundationpose` conda environment


### Launch Files

- **`foundationpose_with_sam.launch`** - Main launch file (SAM + YOLO)
  - Launches SAM segmentation node and FoundationPose node
  - Loads configuration from `config/foundationpose_config.yaml`

- **`foundationpose_with_grounded_sam.launch`** - Grounded SAM version
  - Launches Grounded SAM segmentation node and FoundationPose node
  - Better for open-vocabulary object detection

- **`foundationpose_pose_estimation.launch`** - FoundationPose only
  - Launches FoundationPose node only
  - Requires external mask topic

- **`opencv_pose_estimation.launch`** - OpenCV-based pose estimation (alternative, stashed)
- **`dope_final_project.launch`** - DOPE pose estimation (alternative)

### Configuration

- **`config/foundationpose_config.yaml`** - Main configuration file
  - All parameters organized in one place
  - Object configuration (name, mesh path)
  - Camera topics (RGB, depth, camera info)
  - FoundationPose parameters (refinement iterations, debug level)
  - SAM/Grounded SAM parameters (model type, checkpoint paths, strategies)
  - YOLO parameters (model path, confidence thresholds)
  - Pose quality thresholds
  - Coordinate frame correction settings

- **`config/dope_config.yaml`** - DOPE configuration (if using DOPE)

### Documentation

- **`readmes/README_FOUNDATIONPOSE_SETUP.md`** - Complete FoundationPose setup and usage guide
- **`readmes/README_SAM_SETUP.md`** - SAM installation guide
- **`readmes/README_GROUNDED_SAM_SETUP.md`** - Grounded SAM installation guide
- **`docs/POSE_ESTIMATION_IMPROVEMENTS.md`** - Pose estimation accuracy analysis and improvements

### Development Tools

Located in `tools/` (optional, not required for main pipeline):
- `analyze_mesh_simple.py` - Analyze mesh file properties
- `generate_model_from_image.py` - Generate 3D model from image
- `generate_model_from_mesh.py` - Generate model from mesh file
- `generate_mustard_model.py` - Generate mustard bottle model

These are standalone development/testing tools and are not used by the main system.

### Metrics and Analysis

- **`metrics/plot_metrics.py`** - Generate performance comparison plots
  - Creates distribution plots and bar charts comparing all implementations
  - Requires: `matplotlib`, `numpy`, `pandas`
  - Run after collecting metrics: `python3 metrics/plot_metrics.py`

## Key Features

### Pose Estimation
- **All parameters in one config file** - No need to pass launch arguments
- **Pose quality checking** - Only publishes poses when position error < threshold and orientation error < threshold (configurable)
- **Automatic pose stability detection** - Waits for pose to converge before publishing
- **Coordinate frame correction** - Configurable rotation to fix orientation issues
- **Depth-based mask fallback** - Uses depth information if segmentation mask unavailable
- **Multiple refinement iterations** - Configurable for accuracy vs. speed trade-off

### Segmentation Options

#### SAM with YOLO (Default)
1. **`detection`** (Recommended) - Uses YOLO to detect objects automatically
   - Supports COCO dataset classes (bottle, book, cup, etc.)
   - Automatic class mapping based on object name
2. **`center_point`** - Uses center of image (may fail if object not centered)
3. **`point`** - Uses specific point coordinates
4. **`box`** - Uses bounding box coordinates
5. **`automatic`** - Generates all masks and filters by size

#### Grounded SAM (Alternative)
- **Open-vocabulary detection** - No need for class name mapping
- **Text-based prompts** - Use natural language (e.g., "cracker box", "mustard bottle")
- **Better for custom objects** - Works with objects not in COCO dataset
- **Single integrated solution** - Grounding DINO + SAM in one pipeline


## Requirements

### System Requirements
- **ROS1 Noetic**
- **NVIDIA GPU with CUDA support**
- **Python 3.9** (for conda environments)

### Conda Environments
- **`sam`** - SAM segmentation with YOLO
  - PyTorch, torchvision
  - SAM (Segment Anything Model)
  - YOLO (ultralytics)
  - ROS dependencies

- **`grounded_sam`** - Grounded SAM segmentation
  - PyTorch 1.13.1
  - Grounding DINO
  - SAM (Segment Anything Model)
  - ROS dependencies

- **`foundationpose`** - FoundationPose pose estimation
  - FoundationPose dependencies
  - PyTorch, CUDA
  - ROS dependencies

### Object Meshes
- Object mesh files (`.obj` format) in `meshes/{object_name}/mesh.obj`
- Supported objects: `cracker_box`, `mustard_bottle`, or any custom object
- Mesh files should match object dimensions for accurate pose estimation

## File Structure

```
final-project-OfficialBishal/
├── scripts/                          # Main ROS nodes
│   ├── sam_segmentation_node.py      # SAM + YOLO segmentation
│   ├── grounded_sam_segmentation_node.py  # Grounded SAM segmentation
│   ├── foundationpose_pose_estimation_node.py  # FoundationPose pose estimation
│   └── wrappers/                      # Wrapper scripts (conda environment setup)
│       ├── run_sam_segmentation.sh        # Wrapper script for SAM node
│       ├── run_grounded_sam_segmentation.sh  # Wrapper script for Grounded SAM
│       ├── run_foundationpose.sh          # Wrapper script for FoundationPose
│       └── run_pick_and_place.sh          # Wrapper script for pick and place
├── tools/                              # Development tools (optional)
│   ├── analyze_mesh_simple.py
│   ├── generate_model_from_image.py
│   ├── generate_model_from_mesh.py
│   └── generate_mustard_model.py
├── setup/                             # Setup scripts
│   ├── setup_sam.sh                   # Setup SAM environment
│   └── setup_grounded_sam.sh          # Setup Grounded SAM environment
├── launch/                            # ROS launch files
│   ├── foundationpose_with_sam.launch         # Main launch (SAM + YOLO)
│   ├── foundationpose_with_grounded_sam.launch  # Launch (Grounded SAM)
│   ├── foundationpose_pose_estimation.launch  # FoundationPose only
│   ├── opencv_pose_estimation.launch  # OpenCV pose estimation (stashed)
│   └── dope_final_project.launch      # DOPE pose estimation
├── config/                            # Configuration files
│   ├── foundationpose_config.yaml     # Main configuration (all parameters)
│   └── dope_config.yaml               # DOPE configuration
├── meshes/                            # Object mesh files
│   ├── cracker_box/
│   │   └── mesh.obj
│   └── mustard_bottle/
│       └── mesh.obj
├── sam-checkpoints/                   # SAM model checkpoints
│   └── sam_vit_h.pth
├── readmes/                           # Documentation
│   ├── README_FOUNDATIONPOSE_SETUP.md
│   ├── README_SAM_SETUP.md
│   └── README_GROUNDED_SAM_SETUP.md
├── docs/                              # Additional documentation
│   └── POSE_ESTIMATION_IMPROVEMENTS.md
└── debug/                             # Debug output (generated at runtime)
    ├── foundationpose/
    └── track_vis/
```

## Usage Examples

### Example 1: Pose Estimation with SAM + YOLO

```bash
# 1. Setup (one-time)
./setup/setup_sam.sh

# 2. Configure object in config/foundationpose_config.yaml
#    Set object_name: "cracker_box" or "mustard_bottle"

# 3. Launch system
source ~/hsr_robocanes_omniverse/devel/setup.bash
roslaunch final-project-OfficialBishal foundationpose_with_sam.launch

# 4. Check pose topic
rostopic echo /foundationpose_pose_estimation/pose
```

### Example 2: Pose Estimation with Grounded SAM

```bash
# 1. Setup (one-time)
./setup/setup_grounded_sam.sh

# 2. Configure in config/foundationpose_config.yaml
#    Set object_name: "cracker_box"
#    Grounded SAM will auto-generate text prompt: "cracker box"

# 3. Launch system
roslaunch final-project-OfficialBishal foundationpose_with_grounded_sam.launch
```


## Configuration Guide

### Changing Object

Edit `config/foundationpose_config.yaml`:

```yaml
object_name: "cracker_box"  # or "mustard_bottle" or custom object name
```

The system will automatically:
- Load mesh from `meshes/{object_name}/mesh.obj`
- Use mask topic `/segmentation/{object_name}_mask`
- Map to appropriate YOLO classes (if using SAM + YOLO)

### Adjusting Pose Quality Thresholds

```yaml
publish_quality:
  publish_position_error_threshold: 0.2    # meters
  publish_orientation_error_threshold: 45.0  # degrees
```

### Switching Segmentation Strategy

```yaml
sam:
  segmentation_strategy: "detection"  # Options: detection, center_point, point, box, automatic
```

## Troubleshooting

### Pose Estimation Issues
- **Check mask quality**: Ensure segmentation mask excludes table surface
- **Verify mesh file**: Mesh should match object dimensions
- **Adjust quality thresholds**: Lower thresholds if poses not publishing
- **Check camera topics**: Verify RGB-D topics are publishing

### Environment Issues
- **Import errors**: Verify conda environment is activated
- **CUDA errors**: Check GPU availability and CUDA version compatibility
- **ROS topic errors**: Ensure all required topics are publishing

For detailed troubleshooting, see:
- `readmes/README_FOUNDATIONPOSE_SETUP.md`
- `readmes/README_SAM_SETUP.md`
- `readmes/README_GROUNDED_SAM_SETUP.md`
- `docs/POSE_ESTIMATION_IMPROVEMENTS.md`

## Performance Monitoring and Comparative Analysis

The system includes built-in performance monitoring for all components, allowing you to compare different pose estimation methods and build performance comparison tables.

### Available Metrics

Each component automatically prints performance metrics after each operation:

- **⏱️ Time**: Processing time in milliseconds and seconds
- **🎮 GPU Memory**: Allocated, reserved, and total GPU memory usage
- **🎮 GPU Utilization**: GPU compute utilization percentage (if pynvml available)
- **💻 CPU Usage**: Process and system-wide CPU usage percentage
- **💻 Memory Usage**: Process memory usage in MB

### Step-by-Step Guide for Comparative Analysis

#### Step 1: Prepare Your Environment

```bash
# Ensure all dependencies are installed
cd ~/hsr_robocanes_omniverse/src/final-project-OfficialBishal

# Install psutil for CPU monitoring (optional but recommended)
conda activate sam  # or grounded_sam, or foundationpose
pip install psutil

# Install pynvml for GPU utilization monitoring (optional)
pip install nvidia-ml-py3
```

#### Step 2: Configure for Performance Testing

Edit `config/foundationpose_config.yaml` to optimize for speed or accuracy:

```yaml
foundationpose:
  # For faster processing (lower accuracy)
  est_refine_iter: 1
  track_refine_iter: 1
  debug: 0
  
  # For better accuracy (slower processing)
  # est_refine_iter: 3
  # track_refine_iter: 2
  # debug: 1

sam:
  # Use faster model for segmentation
  sam_model_type: "vit_b"  # Fastest: vit_b, Balanced: vit_l, Most accurate: vit_h
```

#### Step 3: Test Method 1 - FoundationPose Only (No Segmentation)

This baseline shows pose estimation performance without segmentation overhead.

```bash
# Terminal 1: Launch Isaac Sim world
# (Your existing world launch command)

# Terminal 2: Launch FoundationPose only
source ~/hsr_robocanes_omniverse/devel/setup.bash
roslaunch final-project-OfficialBishal foundationpose_pose_estimation.launch

# Watch the terminal output for performance metrics
# Look for lines starting with "PERFORMANCE METRICS - REGISTRATION"
```

**What to Record:**
- Time: [from FoundationPose output]
- GPU Memory: [from FoundationPose output]
- GPU Utilization: [from FoundationPose output]
- CPU Usage: [from FoundationPose output]

**Note:** This method uses depth-based masking, which may be less accurate than proper segmentation.

#### Step 4: Test Method 2 - YOLO + SAM + FoundationPose

This tests the full pipeline with YOLO detection and SAM segmentation.

```bash
# Terminal 1: Launch Isaac Sim world
# (Your existing world launch command)

# Terminal 2: Launch YOLO + SAM + FoundationPose
source ~/hsr_robocanes_omniverse/devel/setup.bash
roslaunch final-project-OfficialBishal foundationpose_with_sam.launch

# Watch both terminal outputs:
# - SAM node will print "PERFORMANCE METRICS - YOLO+SAM (detection) SEGMENTATION"
# - FoundationPose node will print "PERFORMANCE METRICS - REGISTRATION"
```

**What to Record:**
- Segmentation Time: [from SAM node output]
- Pose Estimation Time: [from FoundationPose output]
- Total Time: [Segmentation Time + Pose Estimation Time]
- GPU Memory: [maximum from both nodes]
- GPU Utilization: [maximum from both nodes]
- CPU Usage: [maximum from both nodes]

#### Step 5: Test Method 3 - Grounded SAM + FoundationPose

This tests the full pipeline with Grounded SAM (open-vocabulary detection).

```bash
# Terminal 1: Launch Isaac Sim world
# (Your existing world launch command)

# Terminal 2: Launch Grounded SAM + FoundationPose
source ~/hsr_robocanes_omniverse/devel/setup.bash
roslaunch final-project-OfficialBishal foundationpose_with_grounded_sam.launch

# Watch both terminal outputs:
# - Grounded SAM node will print "PERFORMANCE METRICS - GROUNDED SAM SEGMENTATION"
# - FoundationPose node will print "PERFORMANCE METRICS - REGISTRATION"
```

**What to Record:**
- Segmentation Time: [from Grounded SAM node output]
- Pose Estimation Time: [from FoundationPose output]
- Total Time: [Segmentation Time + Pose Estimation Time]
- GPU Memory: [maximum from both nodes]
- GPU Utilization: [maximum from both nodes]
- CPU Usage: [maximum from both nodes]

#### Step 6: Collect Multiple Samples

For accurate comparison, collect multiple samples for each method:

```bash
# Run each method for at least 10-20 pose estimations
# Record the average, minimum, and maximum values
# This accounts for variability in processing time
```

**Tips for Data Collection:**
- Use the same scene/object position for all methods
- Collect data during robot movement (if testing tracking performance)
- Note any outliers or failures
- Record both successful and failed attempts (for reliability metrics)

#### Step 7: Build Comparison Table

Create a table with the following structure:

| Method | Segmentation Time (ms) | Pose Estimation Time (ms) | Total Time (ms) | GPU Memory (GB) | GPU Utilization (%) | CPU Usage (%) | Notes |
|--------|----------------------|-------------------------|----------------|----------------|---------------------|---------------|-------|
| FoundationPose Only | N/A | [avg] | [avg] | [avg] | [avg] | [avg] | Depth-based mask |
| YOLO + SAM + FoundationPose | [avg] | [avg] | [avg] | [max] | [max] | [max] | COCO classes |
| Grounded SAM + FoundationPose | [avg] | [avg] | [avg] | [max] | [max] | [max] | Open-vocabulary |

**Example Table:**

| Method | Segmentation Time (ms) | Pose Estimation Time (ms) | Total Time (ms) | GPU Memory (GB) | GPU Utilization (%) | CPU Usage (%) |
|--------|----------------------|-------------------------|----------------|----------------|---------------------|---------------|
| FoundationPose Only | N/A | 1234.5 | 1234.5 | 2.45 | 85.3 | 45.2 |
| YOLO + SAM + FoundationPose | 456.7 | 1234.5 | 1691.2 | 3.20 | 92.1 | 52.3 |
| Grounded SAM + FoundationPose | 789.2 | 1234.5 | 2023.7 | 3.45 | 88.7 | 48.9 |

#### Step 8: Analyze Results

**Key Metrics to Compare:**

1. **Total Processing Time**: Lower is better for real-time applications
   - FoundationPose Only: Fastest (no segmentation overhead)
   - YOLO + SAM: Moderate (fast detection + segmentation)
   - Grounded SAM: Slower (more complex detection)

2. **GPU Memory Usage**: Important for systems with limited VRAM
   - All methods use similar GPU memory for FoundationPose
   - Grounded SAM may use slightly more memory

3. **GPU Utilization**: Shows how efficiently GPU is used
   - Higher utilization = better GPU usage
   - Lower utilization = room for parallel processing

4. **CPU Usage**: Important for systems with limited CPU
   - Segmentation nodes use CPU for preprocessing
   - FoundationPose primarily uses GPU

5. **Accuracy vs Speed Trade-off**: 
   - More refinement iterations = better accuracy but slower
   - Faster SAM models (vit_b) = faster but potentially less accurate masks

### Performance Optimization Tips

#### For Faster Processing:
```yaml
foundationpose:
  est_refine_iter: 1        # Reduce from 3 to 1
  track_refine_iter: 1      # Reduce from 2 to 1
  debug: 0                  # Disable debug output

sam:
  sam_model_type: "vit_b"   # Use fastest model
```

#### For Better Accuracy:
```yaml
foundationpose:
  est_refine_iter: 3        # Increase iterations
  track_refine_iter: 2      # More tracking refinement
  debug: 1                  # Enable basic debug

sam:
  sam_model_type: "vit_h"   # Use most accurate model
```

#### For Balanced Performance:
```yaml
foundationpose:
  est_refine_iter: 2        # Moderate iterations
  track_refine_iter: 1      # Minimal tracking refinement
  debug: 0                  # No debug overhead

sam:
  sam_model_type: "vit_l"   # Balanced model
```

### Understanding the Output

Each performance metric printout looks like this:

```
================================================================================
PERFORMANCE METRICS - [METHOD NAME]
================================================================================
⏱️  Time: 1234.56 ms (1.235 s)
🎮 GPU Memory: 2.45 GB / 8.00 GB (30.6%)
🎮 GPU Reserved: 3.20 GB (40.0%)
🎮 GPU Utilization: 85.3%
💻 CPU (Process): 45.2%
💻 CPU (System): 12.5%
💻 Memory (Process): 1024.5 MB
================================================================================
```

**Reading the Metrics:**
- **Time**: Total processing time for this operation
- **GPU Memory**: Current allocated memory vs total GPU memory
- **GPU Reserved**: Memory reserved by PyTorch (may be higher than allocated)
- **GPU Utilization**: Percentage of GPU compute units in use (if available)
- **CPU (Process)**: CPU usage by this specific process
- **CPU (System)**: Overall system CPU usage
- **Memory (Process)**: RAM usage by this process

### Troubleshooting Performance Monitoring

**If CPU metrics are missing:**
```bash
# Install psutil
pip install psutil
```

**If GPU utilization is missing:**
```bash
# Install pynvml
pip install nvidia-ml-py3
```

**If metrics seem incorrect:**
- Ensure you're reading from the correct terminal output
- Check that the node is actually processing frames (not idle)
- Verify GPU/CPU resources are not being shared with other processes

### Example Workflow for Presentation

1. **Setup**: Configure all three methods with same parameters
2. **Test 1**: Run FoundationPose only, collect 20 samples
3. **Test 2**: Run YOLO + SAM + FoundationPose, collect 20 samples
4. **Test 3**: Run Grounded SAM + FoundationPose, collect 20 samples
5. **Analyze**: Calculate averages, create comparison table
6. **Present**: Show table with key insights (speed vs accuracy trade-offs)

## Performance

### Accuracy (with proper segmentation mask)
- **Position Error**: < 5 cm
- **Orientation Error**: < 10 degrees
- **Height Error**: < 2 cm (critical for top-down picking)

### Processing Speed (Typical Values)
- **SAM Segmentation**: ~0.5-1.0 seconds per frame (GPU, depends on model)
- **FoundationPose Estimation**: ~0.5-2.0 seconds per frame (GPU, depends on refinement iterations)
- **Grounded SAM**: ~1.0-2.0 seconds per frame (GPU)
- **Total Pipeline**: ~1.5-4.0 seconds per frame (segmentation + pose estimation)

**Note**: Actual performance depends on GPU model, image resolution, refinement iterations, and SAM model type. Use the built-in performance monitoring to get accurate measurements for your specific setup.

### Performance Metrics Collection and Analysis

The system automatically collects performance metrics (processing time, GPU/CPU usage, memory) for all implementations:

- **FoundationPose**: Complete metrics (time, GPU, CPU, memory)
- **Grounded SAM**: Time metrics (GPU/CPU monitoring optional)
- **YOLO+SAM**: Complete metrics (time, GPU, CPU, memory)

Metrics are saved to `metrics/data/*.json` files during execution.

**To generate comparison plots:**

```bash
cd ~/hsr_robocanes_omniverse/src/final-project-OfficialBishal/metrics
python3 plot_metrics.py
```

This generates:
- Distribution plots (histograms) for each metric: `plots/distribution_*.png`
- Comparison bar charts: `plots/comparison_*.png`
- Summary statistics CSV: `data/metrics_summary.csv`

**Note**: Grounded SAM appears in time-based comparisons. GPU/CPU comparisons only include implementations that collect those metrics (FoundationPose and YOLO+SAM).

## References

- **FoundationPose**: `/src/FoundationPose` or `/isaac_ros_pose_estimation`
- **SAM**: https://github.com/facebookresearch/segment-anything
- **Grounded SAM**: https://github.com/IDEA-Research/Grounded-Segment-Anything
- **YOLO**: https://github.com/ultralytics/ultralytics

For detailed setup instructions, see `readmes/README_FOUNDATIONPOSE_SETUP.md`.

## Attempted Features

### Pick and Place

A pick-and-place implementation was attempted but not successfully completed. The following components were developed but encountered issues during testing:

**What Was Attempted:**
- **`pick_and_place_node.py`** - Pick and place execution node
  - Subscribed to pose estimates from FoundationPose
  - Used MoveIt for motion planning
  - Implemented state machine: IDLE → PREPARING → APPROACHING → GRASPING → LIFTING → COMPLETE
  - Calculated top surface position from object center pose
  - Integrated with HSR robot gripper and arm control

**Launch File:**
- **`pick_and_place.launch`** - Launch file for pick and place node

**Configuration:**
- Pick and place parameters were included in `config/foundationpose_config.yaml`:
  - Approach height, grasp height offset
  - Object dimensions
  - Planning timeout settings

**Status:**
- Implementation was attempted but failed during testing
- The node and launch file remain in the package for reference
- Not recommended for use in the current state

**Note:** The pose estimation system works independently and can be used without pick and place functionality.

