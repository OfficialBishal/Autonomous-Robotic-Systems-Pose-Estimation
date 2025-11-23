# FoundationPose + ROS1 Integration - Optimal Setup

## Overview

This setup uses the **Docker container's existing environment** optimally:
- **ROS1** is installed system-wide in the Docker container (`/opt/ros/noetic`)
- **FoundationPose** is in the `foundationpose` conda environment
- Both work together by adding ROS Python packages to the conda environment's PYTHONPATH

## Why This Approach?

1. **No duplication**: We don't install ROS packages in conda (they're not on PyPI anyway)
2. **Uses existing setup**: Leverages Docker's ROS1 installation
3. **Clean separation**: FoundationPose stays in its conda environment
4. **Simple**: Just add ROS Python paths to PYTHONPATH

## How It Works

The wrapper script (`run_foundationpose.sh`) does the following:

1. **Sources ROS setup** - Makes ROS commands and Python packages available
2. **Sources ROS workspace** - Makes `rospack find` work
3. **Activates conda environment** - Gets FoundationPose and its dependencies
4. **Sets PYTHONPATH** - Adds ROS Python packages so conda Python can import them

```bash
# ROS Python packages (system-wide)
/opt/ros/noetic/lib/python3/dist-packages

# FoundationPose (in workspace)
$HOME/hsr_robocanes_omniverse/src/FoundationPose
```

## File Locations

- **FoundationPose**: `/home/csc752/hsr_robocanes_omniverse/src/FoundationPose/`
- **ROS Python packages**: `/opt/ros/noetic/lib/python3/dist-packages/`
- **Conda environment**: `~/.conda/envs/foundationpose/`

## Running the Node

Simply use the launch file - the wrapper script handles everything:

```bash
# Inside Docker container
conda activate foundationpose  # Optional - wrapper does this
source ~/hsr_robocanes_omniverse/devel/setup.bash
roslaunch final-project-OfficialBishal foundationpose_pose_estimation.launch
```

The wrapper script automatically:
- Sources ROS setup
- Activates conda environment  
- Sets up PYTHONPATH correctly
- Runs the node

## Dependencies Installed

In the `foundationpose` conda environment:
- `rospkg` - For ROS package management
- `catkin_pkg` - For catkin package utilities
- All FoundationPose dependencies (PyTorch, pytorch3d, etc.)

## Troubleshooting

### "No module named 'rospy'"
- Make sure ROS is sourced: `source /opt/ros/noetic/setup.bash`
- Check PYTHONPATH includes: `/opt/ros/noetic/lib/python3/dist-packages`

### "No module named 'estimater'"
- Check FoundationPose path: `/home/csc752/hsr_robocanes_omniverse/src/FoundationPose/`
- Verify PYTHONPATH includes this path

### Verify Setup
```bash
source /opt/ros/noetic/setup.bash
source /opt/conda/etc/profile.d/conda.sh
conda activate foundationpose
export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:/home/csc752/hsr_robocanes_omniverse/src/FoundationPose:$PYTHONPATH"
python -c "import rospy; from estimater import FoundationPose; print('✓ Setup works!')"
```

## Benefits of This Approach

✅ **No environment conflicts** - ROS and FoundationPose stay separate  
✅ **Uses existing infrastructure** - No need to reinstall ROS  
✅ **Maintainable** - Clear separation of concerns  
✅ **Portable** - Works in any Docker container with ROS + conda  

