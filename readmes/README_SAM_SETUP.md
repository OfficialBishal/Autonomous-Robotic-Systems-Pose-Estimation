# SAM (Segment Anything Model) Setup Guide

## Overview

This guide sets up SAM (Segment Anything Model) from Meta for object segmentation, which will provide high-quality masks for FoundationPose pose estimation.

**Repository**: https://github.com/facebookresearch/segment-anything

## Prerequisites

- Docker container with ROS1 Noetic
- Conda installed
- NVIDIA GPU with CUDA support
- At least 8GB VRAM (for ViT-H model) or 4GB VRAM (for ViT-B model)

## Installation Steps

### Step 1: Clone SAM Repository

```bash
cd ~/hsr_robocanes_omniverse
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
```

### Step 2: Create Conda Environment for SAM

```bash
# Create new conda environment (separate from foundationpose)
conda create -n sam python=3.9 -y
conda activate sam

# Install PyTorch with CUDA support
# Check your CUDA version first: nvidia-smi
# For CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1/12.8 (CUDA 12.8 is backward compatible with 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install SAM

```bash
# Install SAM package
pip install git+https://github.com/facebookresearch/segment-anything.git

# Or install from local clone
cd ~/hsr_robocanes_omniverse/segment-anything
pip install -e .

# Install optional dependencies for ROS integration
pip install opencv-python pycocotools matplotlib numpy
```

### Step 4: Install ROS Dependencies in SAM Environment

```bash
# Install ROS Python package dependencies
pip install rospkg catkin_pkg
```

### Step 5: Download SAM Checkpoints

You need to download at least one model checkpoint. Choose based on your GPU memory:

**Option A: ViT-H (Most Accurate, ~2.4GB, requires ~8GB VRAM)**
```bash
cd ~/hsr_robocanes_omniverse/segment-anything
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h.pth
```

**Option B: ViT-L (Good Balance, ~1.2GB, requires ~6GB VRAM)**
```bash
cd ~/hsr_robocanes_omniverse/segment-anything
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O checkpoints/sam_vit_l.pth
```

**Option C: ViT-B (Fastest, ~375MB, requires ~4GB VRAM)**
```bash
cd ~/hsr_robocanes_omniverse/segment-anything
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b.pth
```

**Recommended**: Start with ViT-B for testing, then use ViT-H for production if you have enough VRAM.

### Step 6: Verify Installation

```bash
conda activate sam
python -c "from segment_anything import sam_model_registry; print('✓ SAM installed successfully!')"
```

## Configuration

### Set Model Type

Edit the launch file or set ROS parameter:
- `sam_model_type`: `"vit_h"`, `"vit_l"`, or `"vit_b"`
- `sam_checkpoint_path`: Path to checkpoint file (default: `~/hsr_robocanes_omniverse/segment-anything/checkpoints/sam_vit_b.pth`)

## Usage

### Running SAM Segmentation Node

```bash
# Activate SAM environment
conda activate sam
source ~/hsr_robocanes_omniverse/devel/setup.bash

# Run SAM node
rosrun final-project-OfficialBishal sam_segmentation_node.py
```

### Running with FoundationPose

Use the combined launch file:

```bash
roslaunch final-project-OfficialBishal foundationpose_with_sam.launch
```

## Segmentation Strategies

The SAM node supports multiple segmentation strategies:

1. **Point Prompt** (default): Uses center of image as prompt
2. **Box Prompt**: Uses bounding box detection
3. **Automatic**: Generates all masks and filters by size

See the node parameters for configuration options.

## Troubleshooting

### "CUDA out of memory"
- Use smaller model (ViT-B instead of ViT-H)
- Reduce image resolution
- Close other GPU applications

### "No module named 'segment_anything'"
- Verify SAM is installed: `pip list | grep segment-anything`
- Check conda environment is activated
- Reinstall: `pip install git+https://github.com/facebookresearch/segment-anything.git`

### "No module named 'rospy'"
- Install ROS dependencies: `pip install rospkg catkin_pkg`
- Source ROS: `source /opt/ros/noetic/setup.bash`
- Check PYTHONPATH includes: `/opt/ros/noetic/lib/python3/dist-packages`

## File Locations

- **SAM Repository**: `~/hsr_robocanes_omniverse/segment-anything/`
- **Checkpoints**: `~/hsr_robocanes_omniverse/segment-anything/checkpoints/`
- **Conda Environment**: `~/.conda/envs/sam/`
- **ROS Node**: `~/hsr_robocanes_omniverse/src/final-project-OfficialBishal/scripts/sam_segmentation_node.py`

## Next Steps

After setup, see:
- `SAM_SEGMENTATION_SETUP.md` for integration details
- Launch file: `launch/foundationpose_with_sam.launch`

---

*Last Updated: [Current Date]*

