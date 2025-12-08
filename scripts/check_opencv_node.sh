#!/bin/bash
# Diagnostic script to check OpenCV pose estimation node status
# Run this inside the docker container

echo "=== Checking ROS Topics ==="
echo ""
echo "Image topics:"
rostopic list | grep -E "(image|camera_info)" || echo "No image topics found!"
echo ""
echo "Checking if image topic is publishing:"
timeout 2 rostopic hz /hsrb/head_rgbd_sensor/rgb/image_rect_color 2>&1 | head -5 || echo "Topic not publishing or not found"
echo ""
echo "Checking if camera_info topic is publishing:"
timeout 2 rostopic hz /hsrb/head_rgbd_sensor/rgb/camera_info 2>&1 | head -5 || echo "Topic not publishing or not found"
echo ""
echo "=== Checking Node Status ==="
rosnode list | grep opencv || echo "OpenCV node not found in node list"
echo ""
echo "=== Checking Published Poses ==="
timeout 2 rostopic echo /opencv_pose_estimation/pose -n 1 2>&1 | head -10 || echo "No poses being published"
echo ""
echo "=== Node Info ==="
rosnode info /opencv_pose_estimation 2>&1 | head -20 || echo "Cannot get node info"


