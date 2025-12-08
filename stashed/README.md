# Stashed OpenCV Files

This directory contains unused/not working OpenCV-related files that have been moved from the main package directory to keep it clean.

## Directory Structure

The directory structure is preserved to match the original package layout, making it easy to copy files back if needed:

```
stashed/
├── src/                    # C++ source files
├── scripts/                # Python scripts and shell scripts
│   └── utilities/         # Utility scripts
├── launch/                # ROS launch files
├── config/                # Configuration YAML files
├── docs/                  # Documentation files
└── include/               # Header files
    └── opencv_pnp/        # OpenCV PnP library headers
```

## Files Moved

### Source Files
- `src/opencv_model_registration_node.cpp` - OpenCV model registration node
- `src/opencv_pose_estimation_node.cpp` - OpenCV pose estimation node

### Scripts
- `scripts/opencv_model_registration_node.cpp` - Duplicate registration node
- `scripts/check_opencv_node.sh` - OpenCV node checker script
- `scripts/create_opencv_model_registration.md` - Documentation
- `scripts/utilities/create_opencv_model.py` - Model creation utility

### Launch Files
- `launch/opencv_model_registration.launch` - Model registration launch file
- `launch/opencv_pose_estimation.launch` - Pose estimation launch file

### Config Files
- `config/opencv_model_mustard.yml` - Mustard bottle model config
- `config/opencv_model_mustard_registered.yml` - Registered mustard model
- `config/opencv_model.yml` - Generic model config

### Documentation
- `docs/CREATE_OPENCV_MODEL.md` - Model creation guide

### Headers
- `include/opencv_pnp/` - Entire OpenCV PnP library directory

## Why These Files Were Moved

These OpenCV-based pose estimation files were replaced by FoundationPose, which provides better accuracy and performance. The OpenCV approach was:
- Less accurate (PnP-based pose estimation)
- More complex to set up (required manual model registration)
- Not working reliably in the final implementation

## Restoring Files

To restore any of these files, simply copy them back to their original locations:

```bash
# Example: Restore a launch file
cp stashed/launch/opencv_pose_estimation.launch launch/

# Example: Restore the entire include directory
cp -r stashed/include/opencv_pnp include/
```

## Note

The current working implementation uses:
- **FoundationPose** for pose estimation (primary method)
- **Grounded SAM** or **YOLO+SAM** for segmentation
- No OpenCV-based pose estimation is currently used

