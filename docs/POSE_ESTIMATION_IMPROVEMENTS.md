# Pose Estimation Accuracy Improvements

## Current Error Analysis

### Error Breakdown
- **Position Error**: 0.3526 m (35.26 cm) - **Too high for precise picking**
- **Orientation Error**: 26.34 degrees - **Significant rotation error**
- **Height (Z) Error**: ~30 cm (estimated 0.025 m vs GT 0.325 m) - **CRITICAL for top-down picking**

The height error is the most critical issue for picking operations from the top.

## Root Causes Identified

### 1. **Depth-Based Mask Includes Table Pixels** ⚠️ PRIMARY ISSUE

The current implementation uses a depth-based heuristic mask that includes both the object AND the table surface. This causes FoundationPose's `guess_translation()` function to use the **median depth**, which gets biased toward the table surface rather than the object.

**Problem in FoundationPose code:**
```python
def guess_translation(self, depth, mask, K):
    # ...
    zc = np.median(depth[valid])  # ← Uses median, which includes table!
    center = (np.linalg.inv(K)@np.asarray([uc,vc,1]).reshape(3,1))*zc
```

If the mask includes table pixels, the median depth will be closer to the table than the object, causing the ~30cm height error.

### 2. **Depth Sensor Limitations on Tables**

Depth sensors have known issues with objects on tables:
- **Edge artifacts**: Depth discontinuities at object boundaries cause measurement errors
- **Reflective surfaces**: Tables can cause specular reflections that confuse depth sensors
- **Occlusion shadows**: Parts of objects may have invalid depth readings
- **Mixed pixels**: At object edges, pixels contain both object and background depth

### 3. **Missing Proper Segmentation Mask**

FoundationPose works best with a proper segmentation mask that excludes the table. The current depth-based mask is a fallback that doesn't work well for objects on tables.

## Recommended Solutions

### Priority 1: Use Proper Segmentation Mask ⭐ **HIGHEST IMPACT**

**Action**: Replace depth-based mask with real segmentation mask.

**Options:**
1. **Isaac Sim Segmentation**: Check if Isaac Sim publishes segmentation topics
   - Look for topics like `/rt_detr_segmentation` or similar
   - Enable segmentation in Isaac Sim scene

2. **Segmentation Model**: Use a segmentation model (RT-DETR, SAM, etc.)
   - Subscribe to segmentation topic
   - Set `use_mask: true` in launch file
   - Provide `mask_topic` parameter

**Implementation:**
```python
# In launch file:
<param name="use_mask" value="true"/>
<param name="mask_topic" value="/segmentation/mask"/>

# This will exclude table pixels from the mask
```

**Expected Improvement**: Should reduce height error from ~30cm to <5cm.

---

### Priority 2: Improve Depth-Based Mask (If Segmentation Unavailable)

If proper segmentation is not available, improve the depth-based mask:

#### Option A: Use Percentile Instead of Median
```python
# Find the closest depth (object top) rather than median (might be table)
# In FoundationPose's guess_translation, use:
zc = np.percentile(depth[valid], 10)  # 10th percentile = closest points (object top)
```

#### Option B: Filter Out Table Plane
```python
# Use RANSAC to detect table plane and exclude those pixels
from sklearn.linear_model import RANSACRegressor
import numpy as np

def remove_table_plane(depth_image, mask, K):
    """Detect and remove table plane from mask."""
    # Convert depth to point cloud
    xyz_map = depth2xyzmap(depth_image, K)
    valid_points = xyz_map[mask]
    
    # Use RANSAC to find table plane (largest plane)
    # Remove points within threshold of plane
    # Return refined mask
    pass
```

#### Option C: Use Depth Gradient to Find Object Edges
```python
# Objects on tables have sharp depth discontinuities
# Use depth gradient to find object boundaries
import cv2

def find_object_by_gradient(depth_image, depth_min, depth_max):
    """Find object mask using depth gradient."""
    # Filter by depth range
    depth_range = (depth_image > depth_min) & (depth_image < depth_max)
    
    # Compute depth gradient
    grad_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Find regions with high gradient (object edges)
    # Combine with depth range
    mask = depth_range & (gradient_magnitude > threshold)
    return mask
```

---

### Priority 3: Increase Refinement Iterations

For better accuracy, increase refinement iterations:

**In launch file:**
```xml
<param name="est_refine_iter" value="10"/>  <!-- Instead of 5 -->
<param name="track_refine_iter" value="5"/>  <!-- Instead of 2 -->
```

**Trade-off**: More iterations = better accuracy but slower processing.

---

### Priority 4: Optimize Camera Positioning

**Recommendations:**
- **Move camera closer**: Reduces depth measurement uncertainty
- **More top-down view**: Reduces depth ambiguity at object edges
- **Ensure object fills image**: More pixels = better pose estimation
- **Avoid extreme angles**: Reduces perspective distortion

---

### Priority 5: Enhanced Depth Preprocessing

Add table plane removal before passing to FoundationPose:

```python
def preprocess_depth_for_table_objects(depth_image, K, table_height_threshold=0.05):
    """
    Preprocess depth image to remove table plane.
    
    Args:
        depth_image: Depth image in meters
        K: Camera intrinsic matrix
        table_height_threshold: Maximum height variation for table plane (meters)
    
    Returns:
        Processed depth image with table plane removed
    """
    # 1. Convert to point cloud
    xyz_map = depth2xyzmap(depth_image, K)
    valid_mask = depth_image > 0.001
    
    # 2. Detect table plane using RANSAC
    # 3. Remove points within threshold of table plane
    # 4. Return processed depth
    
    # This ensures FoundationPose only sees object points
    pass
```

---

### Priority 6: Multi-View Fusion

**Strategy**: Capture from multiple angles and fuse estimates:
- Take multiple images from different viewpoints
- Estimate pose from each view
- Fuse using weighted average or RANSAC
- More robust to single-view errors

---

## Immediate Action Plan

### Step 1: Check for Segmentation Availability
```bash
# Check available topics
rostopic list | grep -i segment
rostopic list | grep -i mask

# If segmentation exists, enable it in launch file
```

### Step 2: If Segmentation Available
1. Enable `use_mask: true` in launch file
2. Set `mask_topic` to segmentation topic
3. Test and verify improved accuracy

### Step 3: If Segmentation NOT Available
1. Implement table plane removal
2. Use percentile-based depth estimation
3. Add depth gradient filtering
4. Increase refinement iterations

### Step 4: Verify Improvements
- Check position error (should be <5cm)
- Check orientation error (should be <10 degrees)
- Verify height accuracy for picking

---

## Code Modifications Needed

### 1. Table Plane Removal (if no segmentation)
Location: `process_pose_estimation()` method
- Add RANSAC-based table plane detection
- Filter mask to exclude table pixels
- Use percentile instead of median for height

### 2. Enhanced Mask Validation
Location: `process_pose_estimation()` method
- Add mask quality metrics
- Log mask statistics (pixel count, depth range, etc.)
- Warn if mask likely includes table

### 3. Debug Visualization
Location: Add debug mode
- Visualize mask overlay on RGB image
- Show depth histogram
- Display point cloud with table plane highlighted

---

## Expected Results

### With Proper Segmentation Mask:
- **Position Error**: < 5 cm (from 35 cm)
- **Orientation Error**: < 10 degrees (from 26 degrees)
- **Height Error**: < 2 cm (from 30 cm)
- **Suitable for**: Precise top-down picking ✅

### With Improved Depth Mask:
- **Position Error**: < 10 cm (from 35 cm)
- **Orientation Error**: < 15 degrees (from 26 degrees)
- **Height Error**: < 5 cm (from 30 cm)
- **Suitable for**: Approximate picking, may need refinement

---

## Testing Checklist

- [ ] Verify segmentation topic exists and publishes masks
- [ ] Test with proper segmentation mask
- [ ] If no segmentation, implement table plane removal
- [ ] Increase refinement iterations
- [ ] Verify camera calibration accuracy
- [ ] Test from multiple camera positions
- [ ] Compare error metrics before/after improvements
- [ ] Validate picking accuracy in real scenario

---

## References

- FoundationPose Paper: [Link if available]
- FoundationPose GitHub: `/src/FoundationPose`
- Isaac ROS FoundationPose: `/isaac_ros_pose_estimation`
- Depth Sensor Best Practices: [Add references]

---

## Notes

- The most critical issue is the **height (Z) error** for top-down picking
- **Proper segmentation mask** is the highest-impact solution
- Depth-based masks are inherently limited for objects on tables
- FoundationPose's `guess_translation()` uses median depth, which fails when mask includes table

---

*Last Updated: [Current Date]*
*Status: Analysis Complete - Implementation Pending*

