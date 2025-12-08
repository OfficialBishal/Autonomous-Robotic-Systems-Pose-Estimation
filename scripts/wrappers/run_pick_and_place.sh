#!/bin/bash
# Wrapper script to run Pick and Place node
# Uses system ROS environment (NOT conda) to avoid libffi conflicts with MoveIt

# Deactivate conda if it's active and clear all conda-related environment variables
# This must be done FIRST before sourcing ROS
if [ -n "$CONDA_DEFAULT_ENV" ] || [ -n "$CONDA_PREFIX" ]; then
    # Deactivate conda if it's active
    if command -v conda >/dev/null 2>&1; then
        # Source conda deactivate function if available
        if [ -f "$HOME/.bashrc" ]; then
            source "$HOME/.bashrc" 2>/dev/null || true
        fi
        # Try to deactivate
        conda deactivate 2>/dev/null || true
    fi
fi

# Clear all conda-related environment variables
unset CONDA_DEFAULT_ENV
unset CONDA_PREFIX
unset CONDA_PROMPT_MODIFIER
unset CONDA_PYTHON_EXE
unset CONDA_SHLVL

# Clear LD_PRELOAD completely to avoid conda libffi conflicts
unset LD_PRELOAD

# Remove any conda paths from LD_LIBRARY_PATH
if [ -n "$LD_LIBRARY_PATH" ]; then
    # Remove conda paths from LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v conda | grep -v "$HOME/.conda" | grep -v "/.conda" | tr '\n' ':' | sed 's/:$//' | sed 's/^://')
fi

# Remove conda paths from PATH
if [ -n "$PATH" ]; then
    export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v conda | grep -v "$HOME/.conda" | grep -v "/.conda" | tr '\n' ':' | sed 's/:$//' | sed 's/^://')
fi

# Remove conda paths from PYTHONPATH
if [ -n "$PYTHONPATH" ]; then
    export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v conda | grep -v "$HOME/.conda" | grep -v "/.conda" | tr '\n' ':' | sed 's/:$//' | sed 's/^://')
fi

# Source ROS setup (required for ROS Python packages and rospack)
# When running under roslaunch, ROS is usually already sourced, but we try to source it anyway
# Try multiple possible ROS locations
if [ -z "$ROS_DISTRO" ]; then
    # ROS not sourced yet, try to source it
    if [ -f /opt/ros/noetic/setup.bash ]; then
        source /opt/ros/noetic/setup.bash
    elif [ -f /opt/ros/setup.bash ]; then
        source /opt/ros/setup.bash
    else
        # Try to find setup.bash
        ROS_SETUP=$(find /opt/ros -name "setup.bash" 2>/dev/null | head -n 1)
        if [ -n "$ROS_SETUP" ]; then
            source "$ROS_SETUP"
        fi
    fi
fi

# Source ROS workspace (required for rospack find)
# Try using alias/function 's' if available, otherwise try direct sourcing
if type s >/dev/null 2>&1; then
    # Use alias/function 's' if it exists (likely sources workspace setup)
    s
else
    # Fallback to direct sourcing
    if [ -f ~/hsr_robocanes_omniverse/devel/setup.bash ]; then
        source ~/hsr_robocanes_omniverse/devel/setup.bash
    elif [ -f "$HOME/hsr_robocanes_omniverse/devel/setup.bash" ]; then
        source "$HOME/hsr_robocanes_omniverse/devel/setup.bash"
    elif [ -f "/home/csc752/hsr_robocanes_omniverse/devel/setup.bash" ]; then
        source "/home/csc752/hsr_robocanes_omniverse/devel/setup.bash"
    fi
fi

# Double-check: Ensure LD_PRELOAD is still unset after ROS setup
unset LD_PRELOAD

# Ensure LD_LIBRARY_PATH doesn't have conda paths (check again after ROS setup)
if [ -n "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v conda | grep -v "$HOME/.conda" | grep -v "/.conda" | tr '\n' ':' | sed 's/:$//' | sed 's/^://')
fi

# Get the package path using rospack
# If rospack fails, try to find the package manually
PACKAGE_PATH=$(rospack find final-project-OfficialBishal 2>/dev/null)
if [ -z "$PACKAGE_PATH" ]; then
    # Try to find it manually
    if [ -d "$HOME/hsr_robocanes_omniverse/src/final-project-OfficialBishal" ]; then
        PACKAGE_PATH="$HOME/hsr_robocanes_omniverse/src/final-project-OfficialBishal"
    elif [ -d "/home/csc752/hsr_robocanes_omniverse/src/final-project-OfficialBishal" ]; then
        PACKAGE_PATH="/home/csc752/hsr_robocanes_omniverse/src/final-project-OfficialBishal"
    elif [ -d "$HOME/hsr_robocanes_omniverse/src/final-project-OfficialBishal" ]; then
        PACKAGE_PATH="$HOME/hsr_robocanes_omniverse/src/final-project-OfficialBishal"
    else
        echo "ERROR: Could not find final-project-OfficialBishal package"
        echo "Tried rospack and manual search in common locations"
        exit 1
    fi
fi

# Set PYTHONPATH to include ROS Python packages
# Try to find ROS Python packages directory
if [ -d "/opt/ros/noetic/lib/python3/dist-packages" ]; then
    export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH"
elif [ -d "/opt/ros/lib/python3/dist-packages" ]; then
    export PYTHONPATH="/opt/ros/lib/python3/dist-packages:$PYTHONPATH"
fi

# Debug: Print environment variables
if [ "${DEBUG_PICK_PLACE:-0}" = "1" ] || [ -n "$ROS_MASTER_URI" ]; then
    echo "=== Pick and Place Environment Debug ==="
    echo "CONDA_PREFIX: ${CONDA_PREFIX:-not set}"
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    echo "LD_PRELOAD: ${LD_PRELOAD:-not set}"
    echo "PYTHONPATH: $PYTHONPATH"
    echo "========================================"
fi

# Use system Python explicitly (not conda Python)
# Find system Python3 (usually /usr/bin/python3)
SYSTEM_PYTHON=$(which -a python3 2>/dev/null | grep -v conda | grep -v "$HOME/.conda" | head -n 1)
if [ -z "$SYSTEM_PYTHON" ] || [ ! -f "$SYSTEM_PYTHON" ]; then
    # Fallback to /usr/bin/python3
    SYSTEM_PYTHON="/usr/bin/python3"
fi

# Verify it's not conda Python
if echo "$SYSTEM_PYTHON" | grep -q conda; then
    SYSTEM_PYTHON="/usr/bin/python3"
fi

# Final check: ensure LD_PRELOAD is unset and no conda in PATH
unset LD_PRELOAD
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v conda | grep -v "$HOME/.conda" | tr '\n' ':' | sed 's/:$//' | sed 's/^://')

# Run the node with system Python
# Use env to unset conda variables while preserving ROS environment
exec env -u CONDA_DEFAULT_ENV -u CONDA_PREFIX -u CONDA_PROMPT_MODIFIER -u CONDA_PYTHON_EXE -u CONDA_SHLVL \
    LD_PRELOAD="" \
    "$SYSTEM_PYTHON" "$PACKAGE_PATH/scripts/pick_and_place_node.py" "$@"

