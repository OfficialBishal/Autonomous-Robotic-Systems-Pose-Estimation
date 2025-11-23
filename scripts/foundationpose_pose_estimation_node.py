#!/usr/bin/env python3
"""
FoundationPose ROS Node for 6D Pose Estimation

This node subscribes to RGB and depth camera topics, uses FoundationPose
to estimate object pose, and publishes the results as ROS messages.
It also compares the estimated pose with ground truth from Isaac Sim.
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library
import os
import sys
import ctypes

# ROS
import rospy
import numpy as np
import cv2
import message_filters
import tf2_ros
from tf import TransformListener
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

# Third-party (scipy imported when needed to avoid early import)

# ============================================================================
# LIBFFI CONFIGURATION (Before any libffi-dependent imports)
# ============================================================================

# Set LD_PRELOAD for libffi BEFORE importing cv_bridge
# This must happen before any libraries that depend on libffi are loaded
# Note: LD_PRELOAD must be set before Python starts, but we try here as a fallback
conda_prefix = os.environ.get('CONDA_PREFIX', '')
if conda_prefix:
    libffi_path = os.path.join(conda_prefix, 'lib', 'libffi.so.7')
    if os.path.exists(libffi_path):
        current_preload = os.environ.get('LD_PRELOAD', '')
        if libffi_path not in current_preload:
            os.environ['LD_PRELOAD'] = f"{libffi_path}:{current_preload}" if current_preload else libffi_path
            # Try to force load conda's libffi using dlopen
            try:
                libdl = ctypes.CDLL('libdl.so.2')
                libdl.dlopen(libffi_path, ctypes.RTLD_GLOBAL | ctypes.RTLD_NOW)
            except Exception:
                # If it fails, log but continue - the real fix is in the wrapper script
                pass

# ============================================================================
# FOUNDATIONPOSE PATH SETUP
# ============================================================================

FOUNDATIONPOSE_PATHS = [
    os.path.join(os.path.expanduser('~'), 'hsr_robocanes_omniverse', 'src', 'FoundationPose'),
    os.path.join(os.path.expanduser('~'), 'hsr_robocanes_omniverse', 'FoundationPose'),
    '/home/csc752/hsr_robocanes_omniverse/src/FoundationPose',
]

FOUNDATIONPOSE_PATH = None
for path in FOUNDATIONPOSE_PATHS:
    if os.path.exists(path):
        FOUNDATIONPOSE_PATH = path
        break

if FOUNDATIONPOSE_PATH is None:
    print(f"ERROR: FoundationPose not found. Tried: {FOUNDATIONPOSE_PATHS}")
    sys.exit(1)

sys.path.insert(0, FOUNDATIONPOSE_PATH)

# ============================================================================
# FOUNDATIONPOSE IMPORTS
# ============================================================================

try:
    from estimater import FoundationPose
    from learning.training.predict_score import ScorePredictor
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from Utils import set_seed, set_logging_format
    import trimesh
    import nvdiffrast.torch as dr
except ImportError as e:
    print(f"ERROR: Failed to import FoundationPose modules: {e}")
    print("Make sure you're running in the foundationpose conda environment")
    sys.exit(1)

# ============================================================================
# MAIN NODE CLASS
# ============================================================================

class FoundationPoseNode:
    """
    FoundationPose ROS Node
    
    Subscribes to RGB and depth camera topics, performs 6D pose estimation
    using FoundationPose, and publishes pose estimates and visualizations.
    """
    
    # Constants for visualization
    AXIS_LENGTH = 0.15  # 15cm axis length for coordinate frame markers
    AXIS_COLORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R, G, B for x, y, z axes
    AXIS_NAMES = ['x', 'y', 'z']
    AXIS_SHAFT_DIAMETER = 0.01  # Shaft diameter for arrow markers
    AXIS_HEAD_DIAMETER = 0.02   # Head diameter for arrow markers
    
    # Comparison printing intervals
    COMPARISON_PRINT_INTERVAL = 1.0  # Print comparison every 1 second
    GT_ALL_FRAMES_PRINT_INTERVAL = 5.0  # Print GT in all frames every 5 seconds
    
    def __init__(self):
        """Initialize the FoundationPose node."""
        rospy.init_node('foundationpose_pose_estimation', anonymous=True)
        
        # Set seed for reproducibility (critical for consistent pose estimation)
        # FoundationPose uses random pose hypotheses, so we need deterministic behavior
        set_seed(0)
        # set_logging_format()
        rospy.loginfo("Set random seed to 0 for reproducible pose estimation")
        
        # Load parameters
        self._load_parameters()
        
        # Initialize state variables
        self.camera_K = None
        self.pose_initialized = False
        self.last_pose = None
        self.gt_pose = None
        self.gt_pose_received = False
        
        
        # Setup ROS communication
        self._setup_ros_communication()
        
        # Load mesh and initialize FoundationPose
        self._load_mesh()
        self._initialize_foundationpose()
        
        # Setup subscribers
        self._setup_subscribers()
        
        rospy.loginfo("FoundationPose node initialized")
        rospy.loginfo(f"Subscribing to RGB: {self.rgb_topic}")
        rospy.loginfo(f"Subscribing to Depth: {self.depth_topic}")
        rospy.loginfo(f"Subscribing to CameraInfo: {self.camera_info_topic}")
        rospy.loginfo(f"Subscribing to Ground Truth: /mustard_bottle/ground_truth_pose")
    
    # ========================================================================
    # Initialization Methods
    # ========================================================================
    
    def _load_parameters(self):
        """Load ROS parameters."""
        # Mesh file parameter
        self.mesh_file = rospy.get_param('~mesh_file', '')
        if not self.mesh_file or not os.path.exists(self.mesh_file):
            rospy.logerr(f"Mesh file not found: {self.mesh_file}")
            rospy.logerr("Please set the ~mesh_file parameter to a valid mesh file")
            sys.exit(1)
        
        # Topic parameters
        self.rgb_topic = rospy.get_param('~rgb_topic', '/hsrb/head_rgbd_sensor/rgb/image_rect_color')
        self.depth_topic = rospy.get_param('~depth_topic', '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/hsrb/head_rgbd_sensor/rgb/camera_info')
        self.frame_id = rospy.get_param('~frame_id', 'head_rgbd_sensor_rgb_frame')
        self.object_frame_id = rospy.get_param('~object_frame_id', 'object_pose')
        
        # FoundationPose parameters
        self.est_refine_iter = rospy.get_param('~est_refine_iter', 5)
        self.track_refine_iter = rospy.get_param('~track_refine_iter', 2)
        self.debug = rospy.get_param('~debug', 1)
        
        # Mask parameters
        self.use_mask = rospy.get_param('~use_mask', False)
        self.mask_topic = rospy.get_param('~mask_topic', '')
        
        # Depth mask parameters (for better object isolation)
        self.depth_min = rospy.get_param('~depth_min', 0.3)  # Minimum depth (meters)
        self.depth_max = rospy.get_param('~depth_max', 2.0)  # Maximum depth (meters)
    
    def _setup_ros_communication(self):
        """Setup ROS publishers, subscribers, and TF broadcaster."""
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # TF listener for transforming poses to odom frame (using old tf API for PoseStamped)
        self.tf_listener = TransformListener()
        
        # Publishers
        self.pose_pub = rospy.Publisher('~pose', PoseStamped, queue_size=10)
        self.marker_pub = rospy.Publisher('~markers', MarkerArray, queue_size=10)
        
        # Subscriber for ground truth pose
        self.gt_pose_sub = rospy.Subscriber(
            '/mustard_bottle/ground_truth_pose', 
            PoseStamped, 
            self.gt_pose_callback,
            queue_size=1  # Small queue for latest pose only
        )
        
        # Debug: Check if topic exists after a short delay
        rospy.Timer(rospy.Duration(2.0), self._check_gt_topic, oneshot=True)
    
    def _load_mesh(self):
        """Load the object mesh file."""
        rospy.loginfo(f"Loading mesh from: {self.mesh_file}")
        try:
            self.mesh = trimesh.load(self.mesh_file)
            rospy.loginfo(f"Mesh loaded: {len(self.mesh.vertices)} vertices")
            
            # Check mesh orientation and bounds for debugging
            if self.mesh.vertices.shape[0] > 0:
                min_bounds = self.mesh.vertices.min(axis=0)
                max_bounds = self.mesh.vertices.max(axis=0)
                center = (min_bounds + max_bounds) / 2
                extents = max_bounds - min_bounds
                rospy.loginfo(f"Mesh bounds: min={min_bounds}, max={max_bounds}")
                rospy.loginfo(f"Mesh center: {center}, extents: {extents}")
                rospy.loginfo(f"Mesh extent ratios (X:Y:Z): {extents[0]/extents.max():.2f} : {extents[1]/extents.max():.2f} : {extents[2]/extents.max():.2f}")
        except Exception as e:
            rospy.logerr(f"Failed to load mesh: {e}")
            sys.exit(1)
    
    def _initialize_foundationpose(self):
        """Initialize FoundationPose estimator."""
        rospy.loginfo("Initializing FoundationPose...")
        try:
            self.scorer = ScorePredictor()
            self.refiner = PoseRefinePredictor()
            self.glctx = dr.RasterizeCudaContext()
            
            self.estimator = FoundationPose(
                model_pts=self.mesh.vertices,
                model_normals=self.mesh.vertex_normals,
                mesh=self.mesh,
                scorer=self.scorer,
                refiner=self.refiner,
                glctx=self.glctx,
                debug=self.debug,
                debug_dir=os.path.join(os.path.expanduser('~'), '.ros', 'foundationpose_debug')
            )
            rospy.loginfo("FoundationPose initialized successfully")
        except Exception as e:
            rospy.logerr(f"Failed to initialize FoundationPose: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            sys.exit(1)
    
    def _setup_subscribers(self):
        """Setup synchronized image subscribers."""
        rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        info_sub = message_filters.Subscriber(self.camera_info_topic, CameraInfo)
        
        # Synchronize subscribers
        if self.use_mask and self.mask_topic:
            mask_sub = message_filters.Subscriber(self.mask_topic, Image)
            self.ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub, info_sub, mask_sub], 10)
            self.ts.registerCallback(self.image_callback_with_mask)
        else:
            self.ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub, info_sub], 10)
            self.ts.registerCallback(self.image_callback)
    
    # ========================================================================
    # Image Processing
    # ========================================================================
    
    def extract_camera_matrix(self, camera_info):
        """Extract camera intrinsic matrix K from CameraInfo."""
        K = np.array(camera_info.K).reshape(3, 3)
        return K
    
    def ros_image_to_numpy(self, img_msg, desired_encoding='rgb8'):
        """
        Convert ROS Image message to numpy array without cv_bridge.
        This avoids libffi conflicts between system ROS and conda FoundationPose.
        
        Args:
            img_msg: sensor_msgs.msg.Image
            desired_encoding: Desired output encoding (e.g., 'rgb8', 'bgr8', '32FC1', 'mono8')
        
        Returns:
            numpy.ndarray: Image as numpy array
        """
        height = img_msg.height
        width = img_msg.width
        
        # Convert raw data to numpy array based on encoding
        if img_msg.encoding in ['8UC1', 'mono8']:
            img_array = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(height, width)
            if desired_encoding == 'rgb8':
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_msg.encoding in ['8UC3', 'rgb8', 'bgr8']:
            img_array = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(height, width, 3)
            if img_msg.encoding == 'bgr8' and desired_encoding == 'rgb8':
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            elif img_msg.encoding == 'rgb8' and desired_encoding == 'bgr8':
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif img_msg.encoding in ['16UC1', '16SC1']:
            img_array = np.frombuffer(
                img_msg.data, 
                dtype=np.uint16 if '16UC' in img_msg.encoding else np.int16
            ).reshape(height, width)
            if desired_encoding == '32FC1':
                img_array = img_array.astype(np.float32) / 1000.0  # Convert mm to meters
        elif img_msg.encoding in ['32FC1']:
            img_array = np.frombuffer(img_msg.data, dtype=np.float32).reshape(height, width)
        elif img_msg.encoding in ['32FC3']:
            img_array = np.frombuffer(img_msg.data, dtype=np.float32).reshape(height, width, 3)
        else:
            rospy.logwarn(f"Unsupported encoding: {img_msg.encoding}, attempting default conversion")
            if len(img_msg.data) == height * width:
                img_array = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(height, width)
            elif len(img_msg.data) == height * width * 3:
                img_array = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(height, width, 3)
            else:
                raise ValueError(f"Cannot convert encoding {img_msg.encoding} to {desired_encoding}")
        
        return img_array
    
    def image_callback(self, rgb_msg, depth_msg, info_msg):
        """Callback for synchronized RGB, depth, and camera info."""
        try:
            # Extract camera matrix
            if self.camera_K is None:
                self.camera_K = self.extract_camera_matrix(info_msg)
                rospy.loginfo(f"Camera matrix K:\n{self.camera_K}")
            
            # Convert ROS images to numpy arrays
            rgb_image = self.ros_image_to_numpy(rgb_msg, desired_encoding='rgb8')
            depth_image = self.ros_image_to_numpy(depth_msg, desired_encoding='32FC1')
            
            # Process pose estimation
            self.process_pose_estimation(rgb_image, depth_image, rgb_msg.header)
            
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def image_callback_with_mask(self, rgb_msg, depth_msg, info_msg, mask_msg):
        """Callback with mask for object segmentation."""
        try:
            # Extract camera matrix
            if self.camera_K is None:
                self.camera_K = self.extract_camera_matrix(info_msg)
                rospy.loginfo(f"Camera matrix K:\n{self.camera_K}")
            
            # Convert ROS images to numpy arrays
            rgb_image = self.ros_image_to_numpy(rgb_msg, desired_encoding='rgb8')
            depth_image = self.ros_image_to_numpy(depth_msg, desired_encoding='32FC1')
            mask_image = self.ros_image_to_numpy(mask_msg, desired_encoding='mono8')
            
            # Convert mask to boolean
            ob_mask = mask_image.astype(bool)
            
            # Process pose estimation with mask
            self.process_pose_estimation(rgb_image, depth_image, rgb_msg.header, ob_mask=ob_mask)
            
        except Exception as e:
            rospy.logerr(f"Error in image_callback_with_mask: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    # ========================================================================
    # Pose Estimation
    # ========================================================================
    
    def process_pose_estimation(self, rgb_image, depth_image, header, ob_mask=None):
        """
        Process pose estimation using FoundationPose.
        
        Args:
            rgb_image: RGB image as numpy array
            depth_image: Depth image as numpy array (in meters)
            header: ROS message header
            ob_mask: Optional object mask (boolean array)
        """
        if self.camera_K is None:
            rospy.logwarn("Camera matrix not available yet")
            return
        
        try:
            # Convert depth to meters if needed
            if depth_image.dtype != np.float32:
                depth_image = depth_image.astype(np.float32)
            
            # Create mask if not provided
            if ob_mask is None:
                # Use tighter depth bounds to focus on object (table is at ~0.6m, object is on table)
                valid_depth = (depth_image > self.depth_min) & (depth_image < self.depth_max)
                ob_mask = valid_depth.astype(bool)
                rospy.logwarn_once(f"No mask provided, using depth-based heuristic (depth: {self.depth_min}-{self.depth_max}m). Consider using proper segmentation.")
            
            # Estimate pose
            if not self.pose_initialized:
                rospy.loginfo("Performing initial pose registration...")
                try:
                    pose = self.estimator.register(
                        K=self.camera_K,
                        rgb=rgb_image,
                        depth=depth_image,
                        ob_mask=ob_mask,
                        iteration=self.est_refine_iter
                    )
                    self.pose_initialized = True
                    rospy.loginfo("Pose initialized successfully")
                except Exception as e:
                    rospy.logerr(f"Pose registration failed: {e}")
                    rospy.logwarn("Will retry registration on next frame...")
                    return  # Skip this frame, try again next time
            else:
                # Use tracking (faster, but can drift over time)
                pose = self.estimator.track_one(
                    rgb=rgb_image,
                    depth=depth_image,
                    K=self.camera_K,
                    iteration=self.track_refine_iter
                )
                rospy.loginfo_throttle(5.0, "Tracking pose... (publishing at ~24Hz)")
            
            self.last_pose = pose
            
            # Publish pose and compare with ground truth
            self.publish_pose(pose, header)
            self.compare_with_ground_truth(pose, header)
            self.publish_markers(pose, header)
            
        except Exception as e:
            rospy.logerr(f"Error in pose estimation: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            self.pose_initialized = False
    
    # ========================================================================
    # Pose Publishing
    # ========================================================================
    
    def publish_pose(self, pose, header):
        """
        Publish pose as PoseStamped message and TF transform.
        Transforms pose from camera frame to odom frame for visualization.
        
        Args:
            pose: 4x4 transformation matrix (object in camera frame)
            header: ROS message header
        """
        from scipy.spatial.transform import Rotation
        
        # Create PoseStamped message in camera frame using helper method
        pose_msg_camera = self._pose_matrix_to_pose_stamped(pose, header, self.frame_id)
        
        # Extract quaternion for TF transform
        R = pose[:3, :3]
        t = pose[:3, 3]
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # [x, y, z, w]
        
        # Transform pose from camera frame to odom frame for visualization
        # Use the same helper method as publish_markers() for consistency
        pose_msg = self._transform_pose_with_fallback('odom', pose_msg_camera, timeout=0.1)
        if pose_msg is None:
            # If transform fails, publish in camera frame and log warning
            if not hasattr(self, '_tf_transform_warning_logged'):
                rospy.logwarn_throttle(5.0, f"Could not transform pose to odom frame. Publishing in camera frame.")
                self._tf_transform_warning_logged = True
            pose_msg = pose_msg_camera
        
        # Publish transformed pose (in odom frame if transform succeeded, camera frame otherwise)
        self.pose_pub.publish(pose_msg)
        
        # Publish as TF transform (in camera frame - this is the actual object pose)
        transform_tf = TransformStamped()
        transform_tf.header = header
        transform_tf.header.frame_id = self.frame_id
        transform_tf.child_frame_id = self.object_frame_id
        transform_tf.transform.translation.x = t[0]
        transform_tf.transform.translation.y = t[1]
        transform_tf.transform.translation.z = t[2]
        transform_tf.transform.rotation.x = quat[0]
        transform_tf.transform.rotation.y = quat[1]
        transform_tf.transform.rotation.z = quat[2]
        transform_tf.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(transform_tf)
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def _pose_matrix_to_pose_stamped(self, pose_matrix, header, frame_id):
        """
        Convert 4x4 pose matrix to PoseStamped message.
        
        Args:
            pose_matrix: 4x4 transformation matrix
            header: ROS message header
            frame_id: Target frame ID
        
        Returns:
            PoseStamped: Pose message
        """
        from scipy.spatial.transform import Rotation
        
        R = pose_matrix[:3, :3]
        t = pose_matrix[:3, 3]
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # [x, y, z, w]
        
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.position.x = t[0]
        pose_msg.pose.position.y = t[1]
        pose_msg.pose.position.z = t[2]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        return pose_msg
    
    def _extract_pose_arrays(self, pose_stamped):
        """
        Extract position and orientation arrays from PoseStamped.
        
        Args:
            pose_stamped: PoseStamped message
        
        Returns:
            tuple: (position_array, quaternion_array) as numpy arrays
        """
        pos = np.array([
            pose_stamped.pose.position.x,
            pose_stamped.pose.position.y,
            pose_stamped.pose.position.z
        ])
        quat = np.array([
            pose_stamped.pose.orientation.x,
            pose_stamped.pose.orientation.y,
            pose_stamped.pose.orientation.z,
            pose_stamped.pose.orientation.w
        ])
        return pos, quat
    
    def _transform_pose_with_fallback(self, target_frame, pose_msg, timeout=0.5):
        """
        Transform a PoseStamped message to target_frame with fallback logic.
        
        First tries with the message's timestamp, then falls back to current time
        if that fails (e.g., due to extrapolation into past).
        
        Args:
            target_frame: Target frame ID (e.g., 'map', 'odom', 'head_rgbd_sensor_rgb_frame')
            pose_msg: PoseStamped message to transform
            timeout: Timeout duration for waitForTransform (default: 0.5 seconds)
        
        Returns:
            PoseStamped: Transformed pose, or None if transform fails
        """
        from copy import deepcopy
        
        try:
            # First try with the message's timestamp
            self.tf_listener.waitForTransform(
                target_frame,
                pose_msg.header.frame_id,
                pose_msg.header.stamp,
                rospy.Duration(timeout)
            )
            return self.tf_listener.transformPose(target_frame, pose_msg)
        except Exception:
            # If that fails (likely due to extrapolation into past), try with current time
            try:
                pose_msg_latest = deepcopy(pose_msg)
                pose_msg_latest.header.stamp = rospy.Time.now()
                self.tf_listener.waitForTransform(
                    target_frame,
                    pose_msg.header.frame_id,
                    rospy.Time(0),  # Latest available for waitForTransform
                    rospy.Duration(timeout)
                )
                return self.tf_listener.transformPose(target_frame, pose_msg_latest)
            except Exception:
                # Transform failed completely
                return None
    
    def _check_gt_topic(self, event):
        """Check if ground truth topic exists and has publishers."""
        import sys
        try:
            topic = '/mustard_bottle/ground_truth_pose'
            pub_list = rospy.get_published_topics()
            topic_exists = any(t[0] == topic for t in pub_list)
            if topic_exists:
                print(f"[FOUNDATIONPOSE] ✓ Ground truth topic exists: {topic}", file=sys.stderr, flush=True)
            else:
                print(f"[FOUNDATIONPOSE] ✗ Ground truth topic NOT found: {topic}", file=sys.stderr, flush=True)
                print(f"[FOUNDATIONPOSE] Available topics with 'mustard' or 'ground_truth':", file=sys.stderr, flush=True)
                for t, msg_type in pub_list:
                    if 'mustard' in t.lower() or 'ground_truth' in t.lower() or 'pose' in t.lower():
                        print(f"  - {t} ({msg_type})", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[FOUNDATIONPOSE] Error checking topic: {e}", file=sys.stderr, flush=True)
    
    def gt_pose_callback(self, msg):
        """Callback for ground truth pose from Isaac Sim."""
        self.gt_pose = msg
        
        # Always try to transform to camera frame when new GT pose arrives
        # This ensures gt_pose_camera_frame is always up-to-date
        self.gt_pose_camera_frame = self._transform_pose_with_fallback(self.frame_id, msg, timeout=0.5)
        if self.gt_pose_camera_frame is None:
            if not hasattr(self, '_gt_transform_fail_logged'):
                rospy.logwarn_throttle(5.0, f"Could not transform GT pose to camera frame")
                self._gt_transform_fail_logged = True
        
        if not self.gt_pose_received:
            self.gt_pose_received = True
            import sys
            print(f"[FOUNDATIONPOSE] ✓✓✓ Received first ground truth pose from Isaac Sim: "
                  f"pos=[{msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}], "
                  f"orient=[{msg.pose.orientation.x:.3f}, {msg.pose.orientation.y:.3f}, "
                  f"{msg.pose.orientation.z:.3f}, {msg.pose.orientation.w:.3f}]",
                  file=sys.stderr, flush=True)
            rospy.loginfo("Received first ground truth pose from Isaac Sim")
            # Print ground truth pose in different coordinate systems
            self._print_gt_pose_in_all_frames(msg)
    
    def _print_gt_pose_in_all_frames(self, gt_pose_msg):
        """
        Print ground truth pose in different coordinate systems:
        - World/Map frame (as received from Isaac Sim)
        - Camera frame (transformed for comparison with FoundationPose)
        - Odom frame (via TF transform)
        - Grid coordinates (approximate)
        """
        import sys
        
        # Extract pose in world/map frame (as received)
        gt_world_pos = np.array([
            gt_pose_msg.pose.position.x,
            gt_pose_msg.pose.position.y,
            gt_pose_msg.pose.position.z
        ])
        gt_world_quat = np.array([
            gt_pose_msg.pose.orientation.x,
            gt_pose_msg.pose.orientation.y,
            gt_pose_msg.pose.orientation.z,
            gt_pose_msg.pose.orientation.w
        ])
        
        print("\n" + "="*70, file=sys.stderr, flush=True)
        print("GROUND TRUTH POSE - ALL COORDINATE SYSTEMS", file=sys.stderr, flush=True)
        print("="*70, file=sys.stderr, flush=True)
        
        # 1. World/Map frame (as received from Isaac Sim)
        print(f"[FOUNDATIONPOSE] World/Map frame ({gt_pose_msg.header.frame_id}):", file=sys.stderr, flush=True)
        print(f"  Position:    [{gt_world_pos[0]:.4f}, {gt_world_pos[1]:.4f}, {gt_world_pos[2]:.4f}] m", file=sys.stderr, flush=True)
        print(f"  Orientation: [{gt_world_quat[0]:.4f}, {gt_world_quat[1]:.4f}, {gt_world_quat[2]:.4f}, {gt_world_quat[3]:.4f}]", file=sys.stderr, flush=True)
        
        # 2. Transform to camera frame for comparison with FoundationPose
        gt_camera_pose = self._transform_pose_with_fallback(self.frame_id, gt_pose_msg, timeout=1.0)
        if gt_camera_pose is not None:
            gt_camera_pos, gt_camera_quat = self._extract_pose_arrays(gt_camera_pose)
            
            print(f"[FOUNDATIONPOSE] Camera frame ({self.frame_id}) - for comparison:", file=sys.stderr, flush=True)
            print(f"  Position:    [{gt_camera_pos[0]:.4f}, {gt_camera_pos[1]:.4f}, {gt_camera_pos[2]:.4f}] m", file=sys.stderr, flush=True)
            print(f"  Orientation: [{gt_camera_quat[0]:.4f}, {gt_camera_quat[1]:.4f}, {gt_camera_quat[2]:.4f}, {gt_camera_quat[3]:.4f}]", file=sys.stderr, flush=True)
            
            # Store transformed pose for comparison (already done in gt_pose_callback, but update here too)
            self.gt_pose_camera_frame = gt_camera_pose
        else:
            print(f"[FOUNDATIONPOSE] Could not transform to camera frame", file=sys.stderr, flush=True)
        
        # 3. Transform to odom frame
        gt_odom_pose = self._transform_pose_with_fallback('odom', gt_pose_msg, timeout=1.0)
        if gt_odom_pose is not None:
            gt_odom_pos, gt_odom_quat = self._extract_pose_arrays(gt_odom_pose)
            
            print(f"[FOUNDATIONPOSE] Odom frame (transformed via TF):", file=sys.stderr, flush=True)
            print(f"  Position:    [{gt_odom_pos[0]:.4f}, {gt_odom_pos[1]:.4f}, {gt_odom_pos[2]:.4f}] m", file=sys.stderr, flush=True)
            print(f"  Orientation: [{gt_odom_quat[0]:.4f}, {gt_odom_quat[1]:.4f}, {gt_odom_quat[2]:.4f}, {gt_odom_quat[3]:.4f}]", file=sys.stderr, flush=True)
            
            # 4. Grid coordinates (approximate, from odom position)
            grid_x = int(round(gt_odom_pos[0]))
            grid_y = int(round(gt_odom_pos[1]))
            print(f"[FOUNDATIONPOSE] Grid coordinates (approximate from odom):", file=sys.stderr, flush=True)
            print(f"  Grid (x, y): ({grid_x}, {grid_y})", file=sys.stderr, flush=True)
        else:
            print(f"[FOUNDATIONPOSE] Could not transform to odom frame", file=sys.stderr, flush=True)
        
        print("="*70 + "\n", file=sys.stderr, flush=True)
    
    def compare_with_ground_truth(self, estimated_pose, header):
        """
        Compare estimated pose with ground truth and print both.
        
        Args:
            estimated_pose: 4x4 transformation matrix (estimated)
            header: ROS message header
        """
        if self.gt_pose is None:
            # Log periodically that we're waiting for ground truth (not just once)
            if not hasattr(self, '_gt_wait_count'):
                self._gt_wait_count = 0
            self._gt_wait_count += 1
            if self._gt_wait_count % 100 == 1:  # Every ~4 seconds at 24Hz
                import sys
                print(f"[FOUNDATIONPOSE] Still waiting for ground truth pose (checked {self._gt_wait_count} times). "
                      f"Topic: /mustard_bottle/ground_truth_pose", file=sys.stderr, flush=True)
                rospy.logwarn_throttle(5.0, "Waiting for ground truth pose from /mustard_bottle/ground_truth_pose topic...")
            return
        
        from scipy.spatial.transform import Rotation
        
        # Extract estimated pose
        R_est = estimated_pose[:3, :3]
        t_est = estimated_pose[:3, 3]
        rot_est = Rotation.from_matrix(R_est)
        quat_est = rot_est.as_quat()  # [x, y, z, w]
        
        # Extract ground truth pose - use camera frame if available, otherwise try to transform
        # Both poses MUST be in the same frame (camera frame) for accurate comparison
        gt_actual_frame = None
        if hasattr(self, 'gt_pose_camera_frame') and self.gt_pose_camera_frame is not None:
            # Use transformed pose in camera frame (preferred - already transformed)
            gt_pose_for_comparison = self.gt_pose_camera_frame
            gt_actual_frame = self.frame_id  # Camera frame
        else:
            # Try to transform on-the-fly
            gt_pose_for_comparison = self._transform_pose_with_fallback(self.frame_id, self.gt_pose, timeout=0.1)
            if gt_pose_for_comparison is None:
                # If transform fails, we cannot do accurate comparison
                # Log warning and skip comparison
                if not hasattr(self, '_gt_comparison_skip_logged'):
                    rospy.logwarn_throttle(5.0, f"Cannot compare poses: GT transform to camera frame failed. "
                                                f"GT is in '{self.gt_pose.header.frame_id}' but need '{self.frame_id}'")
                    self._gt_comparison_skip_logged = True
                return  # Skip comparison if frames don't match
            gt_actual_frame = self.frame_id  # Successfully transformed to camera frame
        
        # At this point, gt_actual_frame should always be self.frame_id (camera frame)
        # This check is just for safety
        if gt_actual_frame is None:
            rospy.logwarn("gt_actual_frame is None - this should not happen!")
            gt_actual_frame = self.frame_id
        
        t_gt, quat_gt = self._extract_pose_arrays(gt_pose_for_comparison)
        
        # Convert GT quaternion to rotation matrix
        rot_gt = Rotation.from_quat(quat_gt)
        R_gt = rot_gt.as_matrix()
        
        # Calculate errors
        pos_error = np.linalg.norm(t_est - t_gt)
        R_diff = R_est @ R_gt.T
        rot_diff = Rotation.from_matrix(R_diff)
        angle_error_rad = np.linalg.norm(rot_diff.as_rotvec())
        angle_error_deg = angle_error_rad * 180 / np.pi
        
        # Print comparison (use print for visibility, throttled)
        if not hasattr(self, '_last_comparison_time'):
            self._last_comparison_time = 0
        
        current_time = rospy.get_time()
        if current_time - self._last_comparison_time >= self.COMPARISON_PRINT_INTERVAL:
            # Use sys.stderr for immediate visibility (not buffered)
            import sys
            # Get frame IDs for clarity - both should be in camera frame now
            estimated_frame = self.frame_id
            gt_original_frame = self.gt_pose.header.frame_id if self.gt_pose else "unknown"
            
            print("\n" + "="*60, file=sys.stderr, flush=True)
            print("POSE COMPARISON", file=sys.stderr, flush=True)
            print("="*60, file=sys.stderr, flush=True)
            
            # ===== CAMERA FRAME COMPARISON =====
            print(f"--- Camera Frame ({estimated_frame}) ---", file=sys.stderr, flush=True)
            print(f"ESTIMATED (FoundationPose):", file=sys.stderr, flush=True)
            print(f"  Position:    [{t_est[0]:.4f}, {t_est[1]:.4f}, {t_est[2]:.4f}] m", file=sys.stderr, flush=True)
            print(f"  Orientation: [{quat_est[0]:.4f}, {quat_est[1]:.4f}, {quat_est[2]:.4f}, {quat_est[3]:.4f}]", file=sys.stderr, flush=True)
            print(f"GROUND TRUTH (Isaac Sim) - Original frame: {gt_original_frame}, Transformed to: {gt_actual_frame}:", file=sys.stderr, flush=True)
            print(f"  Position:    [{t_gt[0]:.4f}, {t_gt[1]:.4f}, {t_gt[2]:.4f}] m", file=sys.stderr, flush=True)
            print(f"  Orientation: [{quat_gt[0]:.4f}, {quat_gt[1]:.4f}, {quat_gt[2]:.4f}, {quat_gt[3]:.4f}]", file=sys.stderr, flush=True)
            
            # ===== WORLD/MAP FRAME COMPARISON =====
            # Create PoseStamped message from estimated pose (in camera frame) using helper
            est_pose_camera = self._pose_matrix_to_pose_stamped(estimated_pose, header, self.frame_id)
            
            # Transform estimated pose to map frame
            est_pose_map = self._transform_pose_with_fallback('map', est_pose_camera, timeout=0.1)
            
            # Ground truth is already in map frame, but extract it properly
            gt_map_pos = np.array([
                self.gt_pose.pose.position.x,
                self.gt_pose.pose.position.y,
                self.gt_pose.pose.position.z
            ])
            gt_map_quat = np.array([
                self.gt_pose.pose.orientation.x,
                self.gt_pose.pose.orientation.y,
                self.gt_pose.pose.orientation.z,
                self.gt_pose.pose.orientation.w
            ])
            
            print(f"\n--- World/Map Frame (map) ---", file=sys.stderr, flush=True)
            if est_pose_map is not None:
                est_map_pos, est_map_quat = self._extract_pose_arrays(est_pose_map)
                
                print(f"ESTIMATED (FoundationPose):", file=sys.stderr, flush=True)
                print(f"  Position:    [{est_map_pos[0]:.4f}, {est_map_pos[1]:.4f}, {est_map_pos[2]:.4f}] m", file=sys.stderr, flush=True)
                print(f"  Orientation: [{est_map_quat[0]:.4f}, {est_map_quat[1]:.4f}, {est_map_quat[2]:.4f}, {est_map_quat[3]:.4f}]", file=sys.stderr, flush=True)
            else:
                print(f"ESTIMATED (FoundationPose): [Transform to map frame failed]", file=sys.stderr, flush=True)
            
            print(f"GROUND TRUTH (Isaac Sim) - Original frame: {gt_original_frame}:", file=sys.stderr, flush=True)
            print(f"  Position:    [{gt_map_pos[0]:.4f}, {gt_map_pos[1]:.4f}, {gt_map_pos[2]:.4f}] m", file=sys.stderr, flush=True)
            print(f"  Orientation: [{gt_map_quat[0]:.4f}, {gt_map_quat[1]:.4f}, {gt_map_quat[2]:.4f}, {gt_map_quat[3]:.4f}]", file=sys.stderr, flush=True)
            
            # ===== GRID COORDINATES =====
            print(f"\n--- Grid Coordinates ---", file=sys.stderr, flush=True)
            if est_pose_map is not None:
                # Try to get grid from odom first, then fall back to map
                est_pose_odom = self._transform_pose_with_fallback('odom', est_pose_camera, timeout=0.1)
                if est_pose_odom is not None:
                    est_odom_pos, _ = self._extract_pose_arrays(est_pose_odom)
                    est_grid_x = int(round(est_odom_pos[0]))
                    est_grid_y = int(round(est_odom_pos[1]))
                else:
                    est_grid_x = int(round(est_map_pos[0]))
                    est_grid_y = int(round(est_map_pos[1]))
                
                print(f"ESTIMATED (FoundationPose): Grid (x, y) = ({est_grid_x}, {est_grid_y})", file=sys.stderr, flush=True)
            else:
                print(f"ESTIMATED (FoundationPose): [Grid coordinates unavailable]", file=sys.stderr, flush=True)
            
            # Ground truth grid coordinates (from map frame directly)
            gt_grid_x = int(round(gt_map_pos[0]))
            gt_grid_y = int(round(gt_map_pos[1]))
            print(f"GROUND TRUTH (Isaac Sim): Grid (x, y) = ({gt_grid_x}, {gt_grid_y})", file=sys.stderr, flush=True)
            
            # ===== ERROR METRICS =====
            print(f"\n--- Error Metrics (both poses in {gt_actual_frame} frame) ---", file=sys.stderr, flush=True)
            print(f"  Position error:    {pos_error:.4f} m", file=sys.stderr, flush=True)
            print(f"  Orientation error: {angle_error_deg:.2f} degrees", file=sys.stderr, flush=True)
            print("="*60, file=sys.stderr, flush=True)
            
            # Also print ground truth in all coordinate systems periodically
            if not hasattr(self, '_last_gt_all_frames_time'):
                self._last_gt_all_frames_time = 0
            if current_time - self._last_gt_all_frames_time >= self.GT_ALL_FRAMES_PRINT_INTERVAL:
                self._print_gt_pose_in_all_frames(self.gt_pose)
                self._last_gt_all_frames_time = current_time
            
            print("", file=sys.stderr, flush=True)  # Empty line after comparison
            self._last_comparison_time = current_time
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    def publish_markers(self, pose, header):
        """
        Publish visualization markers for the object.
        Transforms marker to odom frame for RViz visualization.
        
        Args:
            pose: 4x4 transformation matrix (object in camera frame)
            header: ROS message header
        """
        from scipy.spatial.transform import Rotation
        
        marker_array = MarkerArray()
        
        # Create PoseStamped message from pose (in camera frame) using helper method
        pose_camera = self._pose_matrix_to_pose_stamped(pose, header, self.frame_id)
        
        # Transform pose to odom frame for RViz visualization (RViz uses odom as fixed frame)
        pose_odom = self._transform_pose_with_fallback('odom', pose_camera, timeout=0.1)
        if pose_odom is None:
            # If transform fails, use camera frame (but log warning)
            rospy.logwarn_throttle(5.0, "Could not transform marker to odom frame, using camera frame")
            pose_odom = pose_camera
        
        # Create mesh marker in odom frame
        marker = Marker()
        marker.header = pose_odom.header
        marker.header.frame_id = pose_odom.header.frame_id  # Use odom frame
        marker.ns = "foundationpose"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose.position.x = pose_odom.pose.position.x
        marker.pose.position.y = pose_odom.pose.position.y
        marker.pose.position.z = pose_odom.pose.position.z
        marker.pose.orientation.x = pose_odom.pose.orientation.x
        marker.pose.orientation.y = pose_odom.pose.orientation.y
        marker.pose.orientation.z = pose_odom.pose.orientation.z
        marker.pose.orientation.w = pose_odom.pose.orientation.w
        
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.a = 0.5
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.mesh_resource = f"file://{self.mesh_file}"
        marker.mesh_use_embedded_materials = True
        
        marker_array.markers.append(marker)
        
        # Add coordinate frame markers (axes) - positioned at object location in odom frame
        axis_length = self.AXIS_LENGTH
        axis_colors = self.AXIS_COLORS
        axis_names = self.AXIS_NAMES
        
        # Get object position and orientation in odom frame
        obj_pos = np.array([
            pose_odom.pose.position.x,
            pose_odom.pose.position.y,
            pose_odom.pose.position.z
        ])
        obj_quat = pose_odom.pose.orientation
        obj_rot = Rotation.from_quat([obj_quat.x, obj_quat.y, obj_quat.z, obj_quat.w])
        obj_R = obj_rot.as_matrix()
        
        # Axis directions in object frame (x, y, z axes)
        axis_dirs_obj = {
            'x': np.array([1, 0, 0]),
            'y': np.array([0, 1, 0]),
            'z': np.array([0, 0, 1])
        }
        
        for i, (color, axis) in enumerate(zip(axis_colors, axis_names)):
            axis_marker = Marker()
            axis_marker.header = pose_odom.header
            axis_marker.header.frame_id = pose_odom.header.frame_id  # Use odom frame
            axis_marker.ns = "foundationpose_axes"
            axis_marker.id = i + 1
            axis_marker.type = Marker.ARROW
            axis_marker.action = Marker.ADD
            
            # Set a valid identity quaternion to avoid "Uninitialized quaternion" warning
            # (Even though we use points, RViz expects a valid quaternion)
            axis_marker.pose.orientation.w = 1.0
            axis_marker.pose.orientation.x = 0.0
            axis_marker.pose.orientation.y = 0.0
            axis_marker.pose.orientation.z = 0.0
            
            # Transform axis direction from object frame to odom frame
            axis_dir_obj = axis_dirs_obj[axis]
            axis_dir_odom = obj_R @ axis_dir_obj
            
            # Normalize direction vector
            axis_dir_odom = axis_dir_odom / np.linalg.norm(axis_dir_odom)
            
            # Calculate start and end points for the arrow
            start_point = obj_pos
            end_point = obj_pos + axis_dir_odom * axis_length
            
            # Set arrow points (start and end)
            axis_marker.points.append(Point(x=start_point[0], y=start_point[1], z=start_point[2]))
            axis_marker.points.append(Point(x=end_point[0], y=end_point[1], z=end_point[2]))
            
            # Set arrow scale (diameter, not length - length is determined by points)
            axis_marker.scale.x = self.AXIS_SHAFT_DIAMETER
            axis_marker.scale.y = self.AXIS_HEAD_DIAMETER
            axis_marker.scale.z = 0.0   # Not used for ARROW with points
            
            # Set color
            axis_marker.color.a = 1.0
            axis_marker.color.r = color[0]
            axis_marker.color.g = color[1]
            axis_marker.color.b = color[2]
            
            marker_array.markers.append(axis_marker)
        
        self.marker_pub.publish(marker_array)
    
    # ========================================================================
    # Main Loop
    # ========================================================================
    
    def run(self):
        """Run the node (blocking)."""
        rospy.spin()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    try:
        node = FoundationPoseNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
