# Grounded SAM Integration Guide

## Overview

This guide explains how to replace YOLO + SAM with **Grounded SAM** (Grounded-Segment-Anything) for better open-vocabulary object detection and segmentation.

**Repository**: https://github.com/IDEA-Research/Grounded-Segment-Anything

**Benefits**:
- Open-vocabulary detection (no need for class name mapping)
- Better detection for objects not in COCO dataset (like "cracker box")
- Single integrated solution (Grounding DINO + SAM)
- Text-based prompts (e.g., "cracker box", "bottle")

## Installation Steps

### Step 1: Create New Conda Environment (Recommended)

**IMPORTANT**: We create a **separate conda environment** (`grounded_sam`) to avoid breaking your existing `sam` environment. This keeps both setups working independently.

**Option A: Automated Setup (Recommended)**

```bash
cd ~/hsr_robocanes_omniverse/src/final-project-OfficialBishal
./setup/setup_grounded_sam.sh
```

**Option B: Manual Setup**

```bash
# Create new conda environment
conda create -n grounded_sam python=3.9 -y
conda activate grounded_sam

# Install PyTorch 1.13.1 (required for Grounding DINO)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install ROS dependencies
pip install rospkg catkin_pkg opencv-python pycocotools matplotlib numpy
```

### Step 2: Clone Grounded SAM Repository

```bash
cd ~/hsr_robocanes_omniverse
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
```

### Step 3: Install Grounding DINO

```bash
conda activate grounded_sam

# Set environment variables for CUDA compilation
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda  # Adjust if your CUDA is elsewhere

# Install Grounding DINO (use --no-build-isolation to avoid PyTorch version conflicts)
cd GroundingDINO
pip install --no-build-isolation -e .
cd ..
```

**If compilation still fails**, you can try installing without CUDA extensions (slower but works):
```bash
# This will compile without CUDA extensions (CPU only, slower)
export BUILD_WITH_CUDA=False
pip install --no-build-isolation -e GroundingDINO
```

### Step 4: Install Additional Dependencies

```bash
conda activate sam
pip install transformers timm addict yapf supervision
```

### Step 5: Download Grounding DINO Checkpoint

```bash
cd ~/hsr_robocanes_omniverse/Grounded-Segment-Anything
mkdir -p checkpoints
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O checkpoints/groundingdino_swint_ogc.pth
```

### Step 6: Verify Installation

```bash
conda activate grounded_sam
python -c "from groundingdino.util.inference import load_model, load_image, predict, annotate; print('Grounding DINO OK')"
python -c "from segment_anything import sam_model_registry; print('SAM OK')"
```

## Integration with ROS Node

The new `grounded_sam_segmentation_node.py` will:
1. Load both Grounding DINO and SAM models
2. Use text prompts based on `object_name` (e.g., "cracker box", "bottle")
3. Detect objects with Grounding DINO
4. Segment with SAM using detected bounding boxes
5. Publish masks (same ROS interface as before)

## Configuration

Update `config/foundationpose_config.yaml`:

```yaml
# Remove YOLO parameters, add Grounding DINO parameters
grounded_sam:
  # Grounding DINO checkpoint path
  groundingdino_checkpoint: "~/hsr_robocanes_omniverse/Grounded-Segment-Anything/checkpoints/groundingdino_swint_ogc.pth"
  
  # Grounding DINO config file
  groundingdino_config: "~/hsr_robocanes_omniverse/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
  
  # Text prompt (will be auto-generated from object_name if empty)
  text_prompt: ""  # e.g., "cracker box" or "bottle"
  
  # Detection thresholds
  box_threshold: 0.3
  text_threshold: 0.25
```

## Troubleshooting

### PyTorch Version Issues

If you get compilation errors:
1. **Check PyTorch version**: `python -c "import torch; print(torch.__version__)"`
2. **Should be 1.13.x**: If not, reinstall: `pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116`
3. **Verify CUDA**: `python -c "import torch; print(torch.cuda.is_available())"`
4. **Make sure you're in the right environment**: `conda activate grounded_sam`

### CUDA Compilation Errors

**Note**: The setup script installs Grounding DINO **without CUDA extensions** by default due to CUDA version mismatch (PyTorch 1.13.1 CUDA 11.6 vs system CUDA 12.8). This is intentional and works fine - PyTorch will still use CUDA for inference, just custom CUDA extensions won't be compiled.

If you need CUDA extensions for maximum performance:
1. **Check CUDA_HOME**: `echo $CUDA_HOME` (should point to your CUDA installation)
2. **Manually patch PyTorch**: Run `python setup/patch_pytorch_cuda_check.py` to bypass version check
3. **Reinstall with CUDA**: `export BUILD_WITH_CUDA=True && pip install --no-build-isolation -e GroundingDINO`

### Import Errors

If you get import errors:
1. **Verify installation**: `pip list | grep groundingdino`
2. **Check PYTHONPATH**: Make sure Grounded-Segment-Anything is in path
3. **Reinstall**: `pip uninstall groundingdino -y && pip install --no-build-isolation -e GroundingDINO`

## Next Steps

After installation:
1. The new ROS node will be created: `grounded_sam_segmentation_node.py`
2. Update launch file to use the new node
3. Update config file with Grounding DINO parameters
4. Test with: `roslaunch final-project-OfficialBishal foundationpose_with_grounded_sam.launch`

## References

- Grounded SAM Repository: https://github.com/IDEA-Research/Grounded-Segment-Anything
- Grounding DINO Repository: https://github.com/IDEA-Research/GroundingDINO
- PyTorch Installation: https://pytorch.org/get-started/previous-versions/

