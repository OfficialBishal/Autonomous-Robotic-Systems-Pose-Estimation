# RViz Visualization Guide for OpenCV Pose Estimation

## Quick Setup

1. **Start RViz** (inside Docker):
   ```bash
   rviz -d $(rospack find final-project-OfficialBishal)/rviz/final-project-rviz.rviz
   ```

   Or if RViz is already running, just open the config file:
   - File → Open Config → Navigate to `final-project-OfficialBishal/rviz/final-project-rviz.rviz`

## What You'll See

The RViz config includes two displays for pose visualization:

### 1. **OpenCV Pose Estimation** (Green Arrow)
   - **Topic**: `/opencv_pose_estimation/pose`
   - **Type**: `geometry_msgs/PoseStamped`
   - **Visualization**: Green 3D arrow showing the estimated pose
   - **Frame**: `head_rgbd_sensor_rgb_frame` (or as configured in launch file)

### 2. **OpenCV Visualization Image** (Image Display)
   - **Topic**: `/opencv_pose_estimation/visualization_image`
   - **Type**: `sensor_msgs/Image`
   - **Shows**: Camera image with:
     - Detected ORB keypoints (green dots)
     - Coordinate axes (if pose is found)
     - Feature matches (if any)

## Manual Setup (If Config Doesn't Load)

If you want to add the displays manually:

### Add Pose Display:

1. Click **"Add"** button in RViz
2. Select **"Pose"** from the list
3. In the display properties:
   - **Name**: `OpenCV Pose Estimation`
   - **Topic**: `/opencv_pose_estimation/pose`
   - **Shape**: `Arrow (3D)` or `Arrow`
   - **Color**: Green (0, 255, 0)
   - **Axes Length**: 0.2
   - **Shaft Length**: 0.15

### Add Image Display:

1. Click **"Add"** button in RViz
2. Select **"Image"** from the list
3. In the display properties:
   - **Name**: `OpenCV Visualization`
   - **Topic**: `/opencv_pose_estimation/visualization_image`
   - **Transport Hint**: `raw`

## Important Settings

### Fixed Frame
Make sure the **Fixed Frame** in RViz matches the frame where your pose is published:
- Default: `odom` or `map`
- The pose is published in `head_rgbd_sensor_rgb_frame` by default
- You may need to change Fixed Frame to see the pose, or ensure TF is publishing the transform

### Frame Issues?

If you don't see the pose arrow:
1. Check that the topic is publishing: `rostopic echo /opencv_pose_estimation/pose`
2. Check the frame_id: `rostopic echo /opencv_pose_estimation/pose | grep frame_id`
3. Make sure TF is publishing the transform from the pose frame to the fixed frame
4. Try changing Fixed Frame in RViz to match the pose frame_id

## Topics to Monitor

- **Pose**: `/opencv_pose_estimation/pose` - The estimated pose
- **Visualization**: `/opencv_pose_estimation/visualization_image` - Image with features
- **Debug**: Check ROS logs for matching information

## Troubleshooting

**No pose arrow visible?**
- Check if pose is being published: `rostopic hz /opencv_pose_estimation/pose`
- Check frame_id matches Fixed Frame or TF exists
- Make sure the display is enabled (checkbox checked)

**No visualization image?**
- Check if image is being published: `rostopic hz /opencv_pose_estimation/visualization_image`
- Make sure camera topics are active: `rostopic list | grep camera`

**Pose appears in wrong location?**
- The pose is relative to the camera frame
- Check the frame_id in the pose message
- Ensure proper TF transforms are available

