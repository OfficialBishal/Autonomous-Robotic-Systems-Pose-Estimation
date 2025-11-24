# Final Project - Object Pose Estimation and Manipulation

This package provides 6D pose estimation for object manipulation using FoundationPose and SAM (Segment Anything Model).

## Quick Start

### 1. Setup

```bash
# Setup SAM (includes YOLO for object detection)
cd ~/hsr_robocanes_omniverse/src/final-project-OfficialBishal
./scripts/setup_sam.sh

# Verify FoundationPose is set up
conda activate foundationpose
python -c "from estimater import FoundationPose; print('OK')"
```

### 2. Run the System

```bash
# Launch system (all parameters loaded from config file)
source ~/hsr_robocanes_omniverse/devel/setup.bash
roslaunch final-project-OfficialBishal foundationpose_with_sam.launch
```

**Configuration**: Edit `config/foundationpose_config.yaml` to change any parameters. Default configuration uses YOLO object detection for automatic segmentation.

## Main Components

### Scripts
- `sam_segmentation_node.py` - SAM segmentation node (generates object masks)
- `foundationpose_pose_estimation_node.py` - FoundationPose pose estimation node
- `run_sam_segmentation.sh` - Wrapper script for SAM node
- `run_foundationpose.sh` - Wrapper script for FoundationPose node

### Launch Files
- `foundationpose_with_sam.launch` - Main launch file (runs both nodes, loads config)
- `foundationpose_pose_estimation.launch` - FoundationPose only

### Configuration
- `config/foundationpose_config.yaml` - All parameters organized here (mesh paths, camera topics, SAM settings, FoundationPose settings, quality thresholds, etc.)

### Documentation
- `readmes/README_FOUNDATIONPOSE_SETUP.md` - Complete setup and usage guide
- `readmes/README_SAM_SETUP.md` - SAM installation guide

## Key Features

- All parameters are in one config file (`config/foundationpose_config.yaml`) - no need to pass launch arguments
- Pose quality checking - only publishes poses when position error < 0.2m and orientation error < 45 degrees
- Automatic pose stability detection - waits for pose to converge before doing corrections
- YOLO-based object detection for automatic segmentation

### Segmentation Strategies

1. **`detection`** (Recommended) - Uses YOLO to detect objects automatically
2. **`center_point`** - Uses center of image (may fail if object not centered)
3. **`point`** - Uses specific point coordinates
4. **`box`** - Uses bounding box coordinates
5. **`automatic`** - Generates all masks and filters by size

## Requirements

- ROS1 Noetic
- Conda environments: `sam` and `foundationpose`
- NVIDIA GPU with CUDA support
- Object mesh file (`.obj` format) in `meshes/` directory

## File Structure

```
final-project-OfficialBishal/
├── scripts/              # Main ROS nodes
│   ├── sam_segmentation_node.py
│   ├── foundationpose_pose_estimation_node.py
│   ├── run_sam_segmentation.sh
│   ├── run_foundationpose.sh
│   ├── setup_sam.sh
│   └── utilities/        # Utility scripts (not required for main pipeline)
├── launch/               # ROS launch files
│   └── foundationpose_with_sam.launch  # Main launch (loads config)
├── config/               # Configuration files
│   └── foundationpose_config.yaml  # All parameters here
├── meshes/               # Object mesh files
├── sam-checkpoints/      # SAM model checkpoints
├── readmes/              # Documentation
└── docs/                 # Additional documentation
```

For detailed setup instructions, see `readmes/README_FOUNDATIONPOSE_SETUP.md`.

