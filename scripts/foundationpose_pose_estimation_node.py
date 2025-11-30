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
    from Utils import set_seed, set_logging_format, draw_posed_3d_box, draw_xyz_axis
    import trimesh
    import nvdiffrast.torch as dr
    import imageio
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
    # Standard RGB convention: X=Red, Y=Green, Z=Blue
    AXIS_COLORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Red, Green, Blue for x, y, z axes
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
        # These will be loaded in _load_parameters() after config file is loaded
        
        
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
        # Object name will be set in _load_parameters, log it after
    
    # ========================================================================
    # Initialization Methods
    # ========================================================================
    
    def _load_parameters(self):
        """Load ROS parameters from config file (organized structure)."""
        # Object name parameter (used for topics and default paths)
        self.object_name = rospy.get_param('~object_name', 'mustard_bottle')
        rospy.loginfo(f"Object name: {self.object_name}")
        
        # Mesh file parameter (top level)
        # Default mesh paths based on object name
        workspace_root = os.path.expanduser('~/hsr_robocanes_omniverse')
        meshes_dir = os.path.join(workspace_root, 'src', 'final-project-OfficialBishal', 'meshes')
        default_mesh_paths = [
            # Try in object-specific folder (organized structure) - YCB naming convention
            os.path.join(meshes_dir, self.object_name, 'textured.obj'),  # YCB standard name
            os.path.join(meshes_dir, self.object_name, 'textured_simple.obj'),  # FoundationPose demo naming
            os.path.join(meshes_dir, self.object_name, 'mesh.obj'),  # Generic name
            os.path.join(meshes_dir, self.object_name, f'{self.object_name}.obj'),  # Object name as filename
            # Try exact object name (flat structure) - for backward compatibility
            os.path.join(meshes_dir, f'{self.object_name}.obj'),
            # Try without underscores (flat structure)
            os.path.join(meshes_dir, f'{self.object_name.replace("_", "")}.obj'),
            # Special case: if object_name is "cracker_box", also try "craker_box" (typo in filename)
            os.path.join(meshes_dir, 'craker_box.obj') if 'cracker' in self.object_name.lower() else None,
            # Fallback to FoundationPose demo data
            os.path.join(FOUNDATIONPOSE_PATH, 'demo_data', 'mustard0', 'mesh', 'textured_simple.obj'),
            os.path.join(os.path.expanduser('~'), 'hsr_robocanes_omniverse', 'src', 'FoundationPose', 'demo_data', 'mustard0', 'mesh', 'textured_simple.obj'),
        ]
        # Remove None entries
        default_mesh_paths = [p for p in default_mesh_paths if p is not None]
        
        self.mesh_file = rospy.get_param('~mesh_file', '')
        
        # Resolve path relative to workspace root if needed
        if self.mesh_file:
            # Get workspace root from ROS package path or environment
            workspace_root = None
            if 'ROS_PACKAGE_PATH' in os.environ:
                # Use first path in ROS_PACKAGE_PATH (usually workspace/src)
                package_paths = os.environ['ROS_PACKAGE_PATH'].split(':')
                if package_paths:
                    # Go up from src/ to workspace root
                    src_path = package_paths[0]
                    if src_path.endswith('/src'):
                        workspace_root = os.path.dirname(src_path)
            
            # Fallback to common workspace location
            if not workspace_root:
                workspace_root = os.path.expanduser('~/hsr_robocanes_omniverse')
            
            # If path starts with 'src/', resolve relative to workspace root
            if self.mesh_file.startswith('src/'):
                self.mesh_file = os.path.join(workspace_root, self.mesh_file)
            # Expand user home directory if path starts with ~
            elif self.mesh_file.startswith('~'):
                self.mesh_file = os.path.expanduser(self.mesh_file)
            # If relative path, try resolving relative to workspace root
            elif not os.path.isabs(self.mesh_file):
                potential_path = os.path.join(workspace_root, self.mesh_file)
                if os.path.exists(potential_path):
                    self.mesh_file = potential_path
        
        if not self.mesh_file:
            # Try default paths
            for default_path in default_mesh_paths:
                if os.path.exists(default_path):
                    self.mesh_file = default_path
                    rospy.loginfo(f"No mesh_file parameter provided, using default: {self.mesh_file}")
                    break
        
        if not self.mesh_file or not os.path.exists(self.mesh_file):
            rospy.logerr(f"Mesh file not found: {self.mesh_file}")
            rospy.logerr("Please set the ~mesh_file parameter to a valid mesh file")
            if default_mesh_paths:
                rospy.logerr(f"Suggested default: {default_mesh_paths[0]}")
            sys.exit(1)
        
        # Camera topics (from config file nested structure)
        self.rgb_topic = rospy.get_param('~camera/rgb_topic', rospy.get_param('~rgb_topic', '/hsrb/head_rgbd_sensor/rgb/image_rect_color'))
        self.depth_topic = rospy.get_param('~camera/depth_topic', rospy.get_param('~depth_topic', '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw'))
        self.camera_info_topic = rospy.get_param('~camera/camera_info_topic', rospy.get_param('~camera_info_topic', '/hsrb/head_rgbd_sensor/rgb/camera_info'))
        self.frame_id = rospy.get_param('~camera/frame_id', rospy.get_param('~frame_id', 'head_rgbd_sensor_rgb_frame'))
        self.object_frame_id = rospy.get_param('~object_frame_id', 'object_pose')
        
        # FoundationPose parameters (from config file nested structure)
        self.est_refine_iter = rospy.get_param('~foundationpose/est_refine_iter', rospy.get_param('~est_refine_iter', 5))
        self.track_refine_iter = rospy.get_param('~foundationpose/track_refine_iter', rospy.get_param('~track_refine_iter', 2))
        self.debug = rospy.get_param('~foundationpose/debug', rospy.get_param('~debug', 1))
        
        # Mask parameters (from config file nested structure)
        self.use_mask = rospy.get_param('~mask/use_mask', rospy.get_param('~use_mask', False))
        self.mask_topic = rospy.get_param('~mask/mask_topic', rospy.get_param('~mask_topic', ''))
        # If mask_topic is empty, use default based on object_name
        if not self.mask_topic:
            self.mask_topic = f'/segmentation/{self.object_name}_mask'
            rospy.loginfo(f"No mask_topic specified, using default: {self.mask_topic}")
        
        # Depth mask parameters (from config file nested structure)
        self.depth_min = rospy.get_param('~depth_mask/depth_min', rospy.get_param('~depth_min', 0.3))
        self.depth_max = rospy.get_param('~depth_mask/depth_max', rospy.get_param('~depth_max', 2.0))
        
        # Coordinate frame correction (to fix upside-down objects in RViz)
        self.coord_correction_enabled = rospy.get_param('~coordinate_frame_correction/enabled', False)
        if self.coord_correction_enabled:
            euler_zyx_deg = rospy.get_param('~coordinate_frame_correction/rotation_euler_zyx_degrees', [0, 0, 180])
            from scipy.spatial.transform import Rotation
            # Convert degrees to radians and create rotation matrix
            euler_zyx_rad = np.deg2rad(euler_zyx_deg)
            self.coord_correction_rotation = Rotation.from_euler('zyx', euler_zyx_rad, degrees=False)
            self.coord_correction_matrix = self.coord_correction_rotation.as_matrix()
            # Store original Euler angles for debug output (don't convert back from matrix)
            self.coord_correction_euler_deg = euler_zyx_deg
            rospy.loginfo(f"Coordinate frame correction enabled: Euler ZYX (degrees) = {euler_zyx_deg}")
        else:
            self.coord_correction_matrix = None
            self.coord_correction_euler_deg = None
            rospy.loginfo("Coordinate frame correction disabled")
        
        # Pose quality thresholds for publishing (from config file nested structure)
        self.publish_position_error_threshold = rospy.get_param('~publish_quality/publish_position_error_threshold',
                                                                rospy.get_param('~publish_position_error_threshold', 0.2))
        self.publish_orientation_error_threshold = rospy.get_param('~publish_quality/publish_orientation_error_threshold',
                                                                     rospy.get_param('~publish_orientation_error_threshold', 45.0))
    
    def _setup_ros_communication(self):
        """Setup ROS publishers, subscribers, and TF broadcaster."""
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # TF listener for transforming poses to odom frame (using old tf API for PoseStamped)
        self.tf_listener = TransformListener()
        
        # Publishers
        self.pose_pub = rospy.Publisher('~pose', PoseStamped, queue_size=10)
        self.marker_pub = rospy.Publisher('~markers', MarkerArray, queue_size=10)
        
        # Subscriber for ground truth pose (topic based on object name)
        gt_topic = f'/{self.object_name}/ground_truth_pose'
        rospy.loginfo(f"Subscribing to Ground Truth: {gt_topic}")
        self.gt_pose_sub = rospy.Subscriber(
            gt_topic, 
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
            # Store original mesh before FoundationPose modifies it
            self.mesh_original = trimesh.load(self.mesh_file)
            self.mesh = self.mesh_original.copy()  # FoundationPose will modify this
            rospy.loginfo(f"Mesh loaded: {len(self.mesh.vertices)} vertices")
            
            # Compute to_origin transformation from oriented bounds (same as run_demo.py)
            # This transforms from original mesh coordinate system to oriented bounding box coordinate system
            self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh_original)
            rospy.loginfo(f"Computed to_origin transformation from oriented bounds")
            rospy.loginfo(f"Oriented bounds extents: {extents}")
            
            # Store extents and compute bbox for visualization (same as run_demo.py)
            self.extents = extents
            self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
            
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
            # Clear CUDA cache before initialization to reduce memory fragmentation
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    rospy.loginfo("Cleared CUDA cache before FoundationPose initialization")
            except Exception as e:
                rospy.logwarn(f"Could not clear CUDA cache: {e}")
            
            # Get package path for debug folders (before FoundationPose initialization)
            try:
                import rospkg
                rospack = rospkg.RosPack()
                self.package_path = rospack.get_path('final-project-OfficialBishal')
            except Exception as e:
                rospy.logwarn(f"Could not get package path using rospack: {e}")
                # Fallback to workspace root
                self.package_path = os.path.join(os.path.expanduser('~'), 'hsr_robocanes_omniverse', 'src', 'final-project-OfficialBishal')
            
            # Create debug folder in our package for FoundationPose internal debug
            foundationpose_debug_dir = os.path.join(self.package_path, 'debug', 'foundationpose')
            os.makedirs(foundationpose_debug_dir, exist_ok=True)
            
            # Create track_vis folder for our visualization images
            self.debug_dir = os.path.join(self.package_path, 'debug', 'track_vis')
            os.makedirs(self.debug_dir, exist_ok=True)
            rospy.loginfo(f"FoundationPose debug will be saved to: {foundationpose_debug_dir}")
            rospy.loginfo(f"Track visualization will be saved to: {self.debug_dir}")
            
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
                debug_dir=foundationpose_debug_dir  # Use our project folder
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
            # Convert depth to meters if needed and make a writable copy
            # The array from ROS message might be read-only, so we need to copy it
            if depth_image.dtype != np.float32:
                depth_image = depth_image.astype(np.float32)
            else:
                # Make a writable copy even if dtype is already float32
                depth_image = depth_image.copy()
            
            # Validate and clean depth image (matching FoundationPose expectations)
            # FoundationPose expects invalid depths (< 0.001) to be set to 0
            # Also filter depths beyond max range (FoundationPose uses zfar=np.inf in demo, but we use depth_max)
            invalid_mask = (depth_image < 0.001) | (depth_image >= self.depth_max)
            depth_image[invalid_mask] = 0.0
            
            # Create mask if not provided
            if ob_mask is None:
                # Use tighter depth bounds to focus on object (table is at ~0.6m, object is on table)
                # Only consider valid depths (not zero/invalid)
                valid_depth = (depth_image > self.depth_min) & (depth_image < self.depth_max) & (depth_image >= 0.001)
                ob_mask = valid_depth.astype(bool)
                
                # Validate mask has sufficient pixels
                if ob_mask.sum() < 4:
                    rospy.logwarn_throttle(5.0, f"Mask too small ({ob_mask.sum()} pixels). Depth range may be incorrect or object not visible.")
                else:
                    rospy.logwarn_once(f"No mask provided, using depth-based heuristic (depth: {self.depth_min}-{self.depth_max}m, {ob_mask.sum()} pixels). Consider using proper segmentation.")
            else:
                # Validate provided mask
                if ob_mask.dtype != bool:
                    ob_mask = ob_mask.astype(bool)
                if ob_mask.sum() < 4:
                    rospy.logwarn_throttle(5.0, f"Provided mask too small ({ob_mask.sum()} pixels). Pose estimation may fail.")
            
            # Estimate pose
            # Clear CUDA cache before pose registration to reduce memory fragmentation
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass  # Ignore if torch not available
            
            # Always perform registration (tracking disabled for now)
            rospy.loginfo("Performing pose registration...")
            try:
                pose = self.estimator.register(
                    K=self.camera_K,
                    rgb=rgb_image,
                    depth=depth_image,
                    ob_mask=ob_mask,
                    iteration=self.est_refine_iter
                )
                self.pose_initialized = True
                rospy.loginfo("Pose registration successful")
            except Exception as e:
                rospy.logerr(f"Pose registration failed: {e}")
                # Aggressive memory cleanup after failure
                try:
                    import torch
                    import gc
                    if torch.cuda.is_available():
                        # Clear cache multiple times
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        # Force Python garbage collection
                        gc.collect()
                        torch.cuda.empty_cache()
                        rospy.loginfo("Performed aggressive CUDA memory cleanup")
                except Exception:
                    pass
                rospy.logwarn("Will retry registration on next frame...")
                # Add a delay to let memory free up before retry
                rospy.sleep(1.0)
                return  # Skip this frame, try again next time
            
            # If we reach here, pose was successfully assigned
            self.last_pose = pose
            
            # Apply to_origin transformation to match run_demo.py behavior
            # FoundationPose returns pose in centered mesh coordinate system (center at origin)
            # We need to transform it back to original mesh coordinate system (with oriented bounds alignment)
            # This is the same transformation applied in run_demo.py line 69: center_pose = pose@np.linalg.inv(to_origin)
            # Note: to_origin moves the center of the bounding box to the origin, so inv(to_origin) moves origin back to center
            # The pose position should already be at the center (origin in centered frame)
            pose_corrected = pose @ np.linalg.inv(self.to_origin)
            
            # Debug: Log the to_origin translation to understand the center offset
            if self.debug >= 2 and not hasattr(self, '_to_origin_translation_logged'):
                to_origin_translation = self.to_origin[:3, 3]
                inv_to_origin_translation = np.linalg.inv(self.to_origin)[:3, 3]
                rospy.loginfo(f">>> [MESH_CENTER] to_origin translation (moves center to origin): {to_origin_translation}")
                rospy.loginfo(f">>> [MESH_CENTER] inv(to_origin) translation (moves origin to center): {inv_to_origin_translation}")
                rospy.loginfo(f">>> [MESH_CENTER] Pose position before inv(to_origin): {pose[:3, 3]}")
                rospy.loginfo(f">>> [MESH_CENTER] Pose position after inv(to_origin): {pose_corrected[:3, 3]}")
                rospy.loginfo(f">>> [MESH_CENTER] Mesh extents: {self.extents}")
                self._to_origin_translation_logged = True
            
            # Apply coordinate frame correction in camera frame BEFORE error calculation
            # Check if correction should be applied based on Z-axis direction
            correction_applied_cam = False
            if self.coord_correction_enabled and self.coord_correction_matrix is not None:
                from scipy.spatial.transform import Rotation
                R_cam = pose_corrected[:3, :3]
                
                # Check if blue axis (Z-axis) is pointing down in camera frame
                # Z-axis in object frame is [0, 0, 1], transform to camera frame
                z_axis_obj = np.array([0, 0, 1])
                z_axis_cam = R_cam @ z_axis_obj
                
                # In camera frame, "down" typically means negative Y or we check if Z-axis points downward
                # Check if the Z-axis is pointing more downward than upward (dot product with camera's down direction)
                # Camera frame: X right, Y down, Z forward
                # So "down" in camera frame is negative Y direction [0, -1, 0]
                camera_down = np.array([0, -1, 0])
                z_axis_dot_down = np.dot(z_axis_cam, camera_down)
                
                # Apply correction only if Z-axis is pointing down (dot product > threshold, e.g., > 0.5)
                if z_axis_dot_down > 0.5:
                    # Apply correction: R_corrected = R_corr @ R_cam
                    R_corrected_cam = self.coord_correction_matrix @ R_cam
                    # Update pose_corrected with corrected rotation
                    pose_corrected[:3, :3] = R_corrected_cam
                    # Position remains unchanged
                    correction_applied_cam = True
                # Removed verbose debug prints for camera frame correction
            
            # Store flag for error calculation - if correction wasn't applied in camera frame but will be in odom frame,
            # we need to apply it for accurate error calculation
            self._correction_applied_cam = correction_applied_cam
            
            # NOTE: Coordinate frame correction is also applied AFTER transforming to odom frame
            # in publish_pose() and publish_markers() for RViz visualization consistency
            
            # Debug: Log transformation details (throttled, only once)
            if self.debug >= 2 and not hasattr(self, '_to_origin_debug_logged'):
                from scipy.spatial.transform import Rotation
                R_before = pose[:3, :3]
                R_after = pose_corrected[:3, :3]
                rospy.loginfo("="*60)
                rospy.loginfo("POSE TRANSFORMATION DEBUG")
                rospy.loginfo("="*60)
                rospy.loginfo(f"to_origin transformation:\n{self.to_origin}")
                rospy.loginfo(f"Pose before to_origin (from FoundationPose):\n{pose}")
                rospy.loginfo(f"Pose after to_origin:\n{pose_corrected}")
                rot_before = Rotation.from_matrix(R_before)
                rot_after = Rotation.from_matrix(R_after)
                rospy.loginfo(f"Rotation before (euler ZYX): {rot_before.as_euler('zyx', degrees=True)}")
                rospy.loginfo(f"Rotation after (euler ZYX): {rot_after.as_euler('zyx', degrees=True)}")
                rospy.loginfo("="*60)
                self._to_origin_debug_logged = True

            # Save the visualization with the box (when debug >= 2)
            if self.debug >= 2:
                # Draw 3D box and axes on the image (same as run_demo.py lines 70-71)
                # Note: pose_corrected is already the center_pose (pose @ inv(to_origin))
                # Work in RGB throughout - only convert to BGR temporarily for draw_posed_3d_box (uses cv2.line which expects BGR)
                vis_rgb = rgb_image.copy()
                # Convert to BGR only for draw_posed_3d_box (cv2.line expects BGR)
                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
                vis_bgr = draw_posed_3d_box(self.camera_K, img=vis_bgr, ob_in_cam=pose_corrected, bbox=self.bbox)
                # Convert back to RGB for draw_xyz_axis (expects RGB when is_input_rgb=True, returns RGB)
                vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                vis = draw_xyz_axis(vis_rgb, ob_in_cam=pose_corrected, scale=0.1, K=self.camera_K, thickness=3, transparency=0, is_input_rgb=True)
                
                # Save the visualization image in RGB format
                import time
                timestamp = int(time.time() * 1000)  # Use timestamp as filename
                image_path = os.path.join(self.debug_dir, f'{timestamp}.png')
                imageio.imwrite(image_path, vis)
                rospy.loginfo_throttle(2.0, f"Saved visualization to: {image_path}")
                
                # Debug: Log pose information for orientation investigation
                if not hasattr(self, '_orientation_debug_logged'):
                    from scipy.spatial.transform import Rotation
                    R = pose_corrected[:3, :3]
                    t = pose_corrected[:3, 3]
                    rot = Rotation.from_matrix(R)
                    euler_zyx = rot.as_euler('zyx', degrees=True)
                    quat = rot.as_quat()  # [x, y, z, w]
                    rospy.loginfo("="*60)
                    rospy.loginfo("POSE ORIENTATION DEBUG (track_vis)")
                    rospy.loginfo("="*60)
                    rospy.loginfo(f"Position (camera frame): [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
                    rospy.loginfo(f"Quaternion (camera frame): [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
                    rospy.loginfo(f"Euler ZYX (degrees): [{euler_zyx[0]:.2f}, {euler_zyx[1]:.2f}, {euler_zyx[2]:.2f}]")
                    rospy.loginfo(f"Rotation matrix:\n{R}")
                    rospy.loginfo("="*60)
                    self._orientation_debug_logged = True
            
            # Check pose quality before publishing (only publish if error is acceptable)
            quality_check = self._check_pose_quality(pose_corrected)
            if quality_check is not None:
                pos_error, orient_error = quality_check
                if pos_error <= self.publish_position_error_threshold and orient_error <= self.publish_orientation_error_threshold:
                    # Quality is good, publish pose
                    self.publish_pose(pose_corrected, header)
                    self.publish_markers(pose_corrected, header)
                    rospy.loginfo_throttle(5.0, f"Publishing pose: pos_error={pos_error:.4f}m (threshold={self.publish_position_error_threshold:.4f}m), "
                                                f"orient_error={orient_error:.2f}° (threshold={self.publish_orientation_error_threshold:.2f}°)")
                else:
                    # Quality is not good enough, don't publish
                    rospy.logwarn_throttle(2.0, f"NOT publishing pose (quality too low): pos_error={pos_error:.4f}m (threshold={self.publish_position_error_threshold:.4f}m), "
                                                f"orient_error={orient_error:.2f}° (threshold={self.publish_orientation_error_threshold:.2f}°)")
            else:
                # No ground truth available, publish anyway (for initial setup)
                if not hasattr(self, '_no_gt_publish_warning_logged'):
                    rospy.logwarn("No ground truth available. Publishing pose without quality check.")
                    self._no_gt_publish_warning_logged = True
                self.publish_pose(pose_corrected, header)
                self.publish_markers(pose_corrected, header)
            
            # Always compare with ground truth for logging/debugging
            self.compare_with_ground_truth(pose_corrected, header)
            
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
        
        # Apply coordinate frame correction (same as in publish_markers) for consistency
        # Only apply if blue axis (Z-axis) is pointing down
        if self.coord_correction_enabled and self.coord_correction_matrix is not None:
            import numpy as np
            from scipy.spatial.transform import Rotation
            
            # Convert pose_msg to matrix
            quat_before = np.array([
                pose_msg.pose.orientation.x,
                pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z,
                pose_msg.pose.orientation.w
            ])
            R_odom = Rotation.from_quat(quat_before).as_matrix()
            
            # Check if blue axis (Z-axis) is pointing down in odom frame
            # Z-axis in object frame is [0, 0, 1], transform to odom frame
            z_axis_obj = np.array([0, 0, 1])
            z_axis_odom = R_odom @ z_axis_obj
            
            # In odom frame (ROS convention: X forward, Y left, Z up), "down" is negative Z
            # Check if Z-axis is pointing down (negative Z component)
            z_axis_z_component = z_axis_odom[2]
            
            # Apply correction only if Z-axis is pointing down (z_component < -0.5)
            if z_axis_z_component < -0.5:
                # Apply correction: only rotate orientation, keep position unchanged
                R_corrected = self.coord_correction_matrix @ R_odom
                quat_corrected = Rotation.from_matrix(R_corrected).as_quat()
                
                # Update pose_msg with corrected orientation
                pose_msg.pose.orientation.x = quat_corrected[0]
                pose_msg.pose.orientation.y = quat_corrected[1]
                pose_msg.pose.orientation.z = quat_corrected[2]
                pose_msg.pose.orientation.w = quat_corrected[3]
                # Position remains unchanged
        
        # Publish transformed pose (in odom frame if transform succeeded, camera frame otherwise)
        # This pose includes coordinate frame correction if enabled
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
    
    def _check_pose_quality(self, pose_corrected):
        """
        Check both position and orientation errors compared to ground truth.
        
        Args:
            pose_corrected: Pose after to_origin transformation
            
        Returns:
            tuple: (position_error_m, orientation_error_deg) or None if no ground truth available
        """
        if self.gt_pose is None or not hasattr(self, 'gt_pose_camera_frame') or self.gt_pose_camera_frame is None:
            return None
        
        from scipy.spatial.transform import Rotation
        
        # Get ground truth pose in camera frame
        gt_pose_for_comparison = self.gt_pose_camera_frame
        t_gt, quat_gt = self._extract_pose_arrays(gt_pose_for_comparison)
        rot_gt = Rotation.from_quat(quat_gt)
        R_gt = rot_gt.as_matrix()
        
        # Extract estimated pose
        R_est = pose_corrected[:3, :3]
        t_est = pose_corrected[:3, 3]
        
        # Calculate position error
        pos_error = np.linalg.norm(t_est - t_gt)
        
        # Calculate orientation error
        R_diff = R_est @ R_gt.T
        rot_diff = Rotation.from_matrix(R_diff)
        angle_error_rad = np.linalg.norm(rot_diff.as_rotvec())
        angle_error_deg = angle_error_rad * 180 / np.pi
        
        return (pos_error, angle_error_deg)
    
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
        Also tries intermediate frames (odom, base_link) if direct transform fails.
        
        Args:
            target_frame: Target frame ID (e.g., 'map', 'odom', 'head_rgbd_sensor_rgb_frame')
            pose_msg: PoseStamped message to transform
            timeout: Timeout duration for waitForTransform (default: 0.5 seconds)
        
        Returns:
            PoseStamped: Transformed pose, or None if transform fails
        """
        from copy import deepcopy
        
        source_frame = pose_msg.header.frame_id
        
        # If source and target are the same, return as-is
        if source_frame == target_frame:
            return pose_msg
        
        error_msgs = []
        
        # Try direct transform first
        try:
            # First try with the message's timestamp
            self.tf_listener.waitForTransform(
                target_frame,
                source_frame,
                pose_msg.header.stamp,
                rospy.Duration(timeout)
            )
            return self.tf_listener.transformPose(target_frame, pose_msg)
        except Exception as e1:
            error_msgs.append(f"Direct transform (with timestamp): {str(e1)[:150]}")
            # If that fails, try with current time
            try:
                pose_msg_latest = deepcopy(pose_msg)
                pose_msg_latest.header.stamp = rospy.Time.now()
                self.tf_listener.waitForTransform(
                    target_frame,
                    source_frame,
                    rospy.Time(0),  # Latest available for waitForTransform
                    rospy.Duration(timeout)
                )
                return self.tf_listener.transformPose(target_frame, pose_msg_latest)
            except Exception as e2:
                error_msgs.append(f"Direct transform (latest time): {str(e2)[:150]}")
                # Try multi-hop transform through intermediate frames
                # Common intermediate frames: odom, base_link
                intermediate_frames = ['odom', 'base_link', 'base_footprint']
                
                for intermediate in intermediate_frames:
                    try:
                        # Transform source -> intermediate -> target
                        # Use a small time offset in the past to avoid extrapolation errors
                        # First: source -> intermediate
                        pose_intermediate = deepcopy(pose_msg)
                        # Use a small offset in the past to ensure we don't extrapolate
                        # rospy.Time(0) means "latest", but we need to be slightly in the past
                        latest_time = self.tf_listener.getLatestCommonTime(source_frame, intermediate)
                        pose_intermediate.header.stamp = latest_time
                        
                        self.tf_listener.waitForTransform(
                            intermediate,
                            source_frame,
                            latest_time,
                            rospy.Duration(timeout)
                        )
                        pose_intermediate = self.tf_listener.transformPose(intermediate, pose_intermediate)
                        
                        # Second: intermediate -> target
                        # Get latest common time for the second transform
                        latest_time2 = self.tf_listener.getLatestCommonTime(intermediate, target_frame)
                        pose_intermediate.header.stamp = latest_time2
                        self.tf_listener.waitForTransform(
                            target_frame,
                            intermediate,
                            latest_time2,
                            rospy.Duration(timeout)
                        )
                        result = self.tf_listener.transformPose(target_frame, pose_intermediate)
                        rospy.logdebug(f"Successfully transformed {source_frame} -> {intermediate} -> {target_frame}")
                        return result
                    except Exception as e3:
                        error_msgs.append(f"Multi-hop via {intermediate}: {str(e3)[:150]}")
                        continue  # Try next intermediate frame
                
                # All transforms failed - log detailed error
                error_key = f"{source_frame}->{target_frame}"
                if not hasattr(self, '_tf_error_logged'):
                    self._tf_error_logged = set()
                
                if error_key not in self._tf_error_logged:
                    rospy.logwarn(f"Transform failed: {source_frame} -> {target_frame}")
                    for err in error_msgs[-3:]:  # Show last 3 errors
                        rospy.logwarn(f"  {err}")
                    rospy.logwarn(f"  Tried intermediate frames: {intermediate_frames}")
                    # Check if frames exist in TF tree (old tf API doesn't have getFrameStrings)
                    try:
                        # Try to get all frames using tf API
                        # Note: old tf.TransformListener doesn't have getFrameStrings()
                        # We can check if transform exists by trying a very short wait
                        import tf
                        # Just log that we can't easily query the TF tree with old API
                        rospy.logwarn(f"  Note: Cannot easily query TF tree with old tf API")
                        rospy.logwarn(f"  Try: rosrun tf view_frames (to see TF tree structure)")
                    except Exception as e:
                        rospy.logwarn(f"  Could not query TF tree: {e}")
                    self._tf_error_logged.add(error_key)
                
                return None
    
    def _check_gt_topic(self, event):
        """Check if ground truth topic exists and has publishers."""
        import sys
        try:
            topic = f'/{self.object_name}/ground_truth_pose'
            pub_list = rospy.get_published_topics()
            topic_exists = any(t[0] == topic for t in pub_list)
            if topic_exists:
                print(f"[FOUNDATIONPOSE] Ground truth topic exists: {topic}", file=sys.stderr, flush=True)
            else:
                print(f"[FOUNDATIONPOSE] Ground truth topic NOT found: {topic}", file=sys.stderr, flush=True)
                print(f"[FOUNDATIONPOSE] Available topics with '{self.object_name}' or 'ground_truth':", file=sys.stderr, flush=True)
                for t, msg_type in pub_list:
                    if self.object_name.lower() in t.lower() or 'ground_truth' in t.lower() or 'pose' in t.lower():
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
            print(f"[FOUNDATIONPOSE] Received first ground truth pose from Isaac Sim: "
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
                gt_topic = f'/{self.object_name}/ground_truth_pose'
                print(f"[FOUNDATIONPOSE] Still waiting for ground truth pose (checked {self._gt_wait_count} times). "
                      f"Topic: {gt_topic}", file=sys.stderr, flush=True)
                rospy.logwarn_throttle(5.0, f"Waiting for ground truth pose from {gt_topic} topic...")
            return
        
        from scipy.spatial.transform import Rotation
        
        # Extract estimated pose
        # IMPORTANT: If coordinate frame correction was applied in odom frame but not in camera frame,
        # we need to apply it here for accurate error calculation
        estimated_pose_for_error = estimated_pose.copy()
        if (hasattr(self, '_correction_applied_cam') and not self._correction_applied_cam and 
            self.coord_correction_enabled and self.coord_correction_matrix is not None):
            # Correction wasn't applied in camera frame, but it will be in odom frame
            # Apply it here for consistent error calculation
            R_est_uncorrected = estimated_pose[:3, :3]
            R_est_corrected = self.coord_correction_matrix @ R_est_uncorrected
            estimated_pose_for_error[:3, :3] = R_est_corrected
            # Removed verbose debug print for error calculation correction
        
        R_est = estimated_pose_for_error[:3, :3]
        t_est = estimated_pose_for_error[:3, 3]
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
        
        # Calculate orientation error using rotation matrix difference
        # This handles quaternion sign ambiguity correctly (q and -q represent same rotation)
        R_diff = R_est @ R_gt.T
        rot_diff = Rotation.from_matrix(R_diff)
        angle_error_rad = np.linalg.norm(rot_diff.as_rotvec())
        angle_error_deg = angle_error_rad * 180 / np.pi
        
        # Handle the case where the error is close to 180° - this might indicate
        # that the quaternions have opposite signs but represent the same orientation
        # Check if quaternions are negated versions of each other
        quat_est_normalized = quat_est / np.linalg.norm(quat_est)
        quat_gt_normalized = quat_gt / np.linalg.norm(quat_gt)
        
        # Check both q and -q (quaternion sign ambiguity)
        dot_product_1 = np.abs(np.dot(quat_est_normalized, quat_gt_normalized))
        dot_product_2 = np.abs(np.dot(quat_est_normalized, -quat_gt_normalized))
        max_dot = max(dot_product_1, dot_product_2)
        
        # Removed verbose debug print for quaternion dot products
        
        # If quaternions are nearly opposite (dot product close to 1), the actual error is small
        # The rotation matrix method should handle this, but if angle is ~180°, check quaternion alignment
        # Lower threshold to 0.9 to catch more cases
        if angle_error_deg > 90 and max_dot > 0.9:
            # Quaternions are nearly aligned (possibly with sign flip), recalculate error
            # Use the better-aligned quaternion
            if dot_product_2 > dot_product_1:
                quat_gt_aligned = -quat_gt_normalized
            else:
                quat_gt_aligned = quat_gt_normalized
            
            rot_est = Rotation.from_quat(quat_est_normalized)
            rot_gt_aligned = Rotation.from_quat(quat_gt_aligned)
            R_est_aligned = rot_est.as_matrix()
            R_gt_aligned = rot_gt_aligned.as_matrix()
            R_diff_aligned = R_est_aligned @ R_gt_aligned.T
            rot_diff_aligned = Rotation.from_matrix(R_diff_aligned)
            angle_error_rad_aligned = np.linalg.norm(rot_diff_aligned.as_rotvec())
            angle_error_deg_aligned = angle_error_rad_aligned * 180 / np.pi
            
            # Use the smaller error (should be the correct one)
            if angle_error_deg_aligned < angle_error_deg:
                angle_error_deg = angle_error_deg_aligned
                # Removed verbose debug print for quaternion sign ambiguity
        
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
            
            # Debug: Check if quaternions are nearly opposite (quaternion sign ambiguity)
            quat_est_normalized = quat_est / np.linalg.norm(quat_est)
            quat_gt_normalized = quat_gt / np.linalg.norm(quat_gt)
            dot_product_1 = np.abs(np.dot(quat_est_normalized, quat_gt_normalized))
            dot_product_2 = np.abs(np.dot(quat_est_normalized, -quat_gt_normalized))
            max_dot = max(dot_product_1, dot_product_2)
            
            if angle_error_deg > 90 and max_dot > 0.99:
                print(f"  NOTE: Quaternion sign ambiguity detected (dot product: {max_dot:.4f}). "
                      f"Actual error may be smaller than reported.", file=sys.stderr, flush=True)
            
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
        import numpy as np  # Import at function level to avoid issues
        
        marker_array = MarkerArray()
        
        # Create PoseStamped message from pose (in camera frame) using helper method
        pose_camera = self._pose_matrix_to_pose_stamped(pose, header, self.frame_id)
        
        # Transform pose to odom frame for RViz visualization (RViz uses odom as fixed frame)
        pose_odom = self._transform_pose_with_fallback('odom', pose_camera, timeout=0.1)
        if pose_odom is None:
            # If transform fails, use camera frame (but log warning)
            rospy.logwarn_throttle(5.0, "Could not transform marker to odom frame, using camera frame")
            pose_odom = pose_camera
        
        # Apply coordinate frame correction AFTER transforming to odom frame
        # This ensures the correction is applied in the frame where RViz visualizes (odom)
        if self.coord_correction_enabled and self.coord_correction_matrix is not None:
            from scipy.spatial.transform import Rotation
            # Convert pose_odom to matrix
            pos_odom = np.array([
                pose_odom.pose.position.x,
                pose_odom.pose.position.y,
                pose_odom.pose.position.z
            ])
            quat_odom_before = np.array([
                pose_odom.pose.orientation.x,
                pose_odom.pose.orientation.y,
                pose_odom.pose.orientation.z,
                pose_odom.pose.orientation.w
            ])
            R_odom = Rotation.from_quat(quat_odom_before).as_matrix()
            euler_odom_before = Rotation.from_quat(quat_odom_before).as_euler('zyx', degrees=True)
            
            # Check if blue axis (Z-axis) is pointing down in odom frame
            # Z-axis in object frame is [0, 0, 1], transform to odom frame
            z_axis_obj = np.array([0, 0, 1])
            z_axis_odom = R_odom @ z_axis_obj
            
            # In odom frame (ROS convention: X forward, Y left, Z up), "down" is negative Z
            # Check if Z-axis is pointing down (negative Z component)
            z_axis_z_component = z_axis_odom[2]
            
            # Only apply correction if Z-axis is pointing down
            should_apply_correction = z_axis_z_component < -0.5
            
            # Removed verbose debug prints for BEFORE correction
            
            # Apply correction only if Z-axis is pointing down
            if should_apply_correction:
                # Apply correction: only rotate orientation, keep position unchanged
                R_corrected = self.coord_correction_matrix @ R_odom
                quat_corrected = Rotation.from_matrix(R_corrected).as_quat()
                euler_odom_after = Rotation.from_quat(quat_corrected).as_euler('zyx', degrees=True)
            else:
                # No correction applied, use original values
                quat_corrected = quat_odom_before
                euler_odom_after = euler_odom_before
                R_corrected = R_odom
            
            # Removed verbose debug prints for AFTER correction
            
            # Update pose_odom with corrected orientation (or original if no correction applied)
            pose_odom.pose.orientation.x = quat_corrected[0]
            pose_odom.pose.orientation.y = quat_corrected[1]
            pose_odom.pose.orientation.z = quat_corrected[2]
            pose_odom.pose.orientation.w = quat_corrected[3]
            # Position remains unchanged
        
        # Debug: Log orientation comparison between camera frame (track_vis) and odom frame (RViz)
        if self.debug >= 2 and not hasattr(self, '_rviz_orientation_debug_logged'):
            from scipy.spatial.transform import Rotation
            
            # Camera frame orientation (what's shown in track_vis)
            R_cam = pose[:3, :3]
            rot_cam = Rotation.from_matrix(R_cam)
            quat_cam = rot_cam.as_quat()
            euler_cam = rot_cam.as_euler('zyx', degrees=True)
            
            # Odom frame orientation (what's shown in RViz)
            quat_odom = pose_odom.pose.orientation
            rot_odom = Rotation.from_quat([quat_odom.x, quat_odom.y, quat_odom.z, quat_odom.w])
            euler_odom = rot_odom.as_euler('zyx', degrees=True)
            
            # Get the camera-to-odom transform to understand the rotation difference
            try:
                latest_time = self.tf_listener.getLatestCommonTime(self.frame_id, 'odom')
                self.tf_listener.waitForTransform('odom', self.frame_id, latest_time, rospy.Duration(0.1))
                (trans_cam_to_odom, rot_cam_to_odom) = self.tf_listener.lookupTransform('odom', self.frame_id, latest_time)
                rot_cam_to_odom_obj = Rotation.from_quat([rot_cam_to_odom[0], rot_cam_to_odom[1], rot_cam_to_odom[2], rot_cam_to_odom[3]])
                euler_cam_to_odom = rot_cam_to_odom_obj.as_euler('zyx', degrees=True)
                R_cam_to_odom = rot_cam_to_odom_obj.as_matrix()
            except Exception as e:
                rospy.logwarn(f"Could not get camera-to-odom transform: {e}")
                R_cam_to_odom = None
                euler_cam_to_odom = None
            
            rospy.loginfo("="*60)
            rospy.loginfo("ORIENTATION COMPARISON: track_vis (camera) vs RViz (odom)")
            rospy.loginfo("="*60)
            rospy.loginfo("CAMERA FRAME (track_vis):")
            rospy.loginfo(f"  Quaternion: [{quat_cam[0]:.4f}, {quat_cam[1]:.4f}, {quat_cam[2]:.4f}, {quat_cam[3]:.4f}]")
            rospy.loginfo(f"  Euler ZYX (degrees): [{euler_cam[0]:.2f}, {euler_cam[1]:.2f}, {euler_cam[2]:.2f}]")
            rospy.loginfo("ODOM FRAME (RViz):")
            rospy.loginfo(f"  Quaternion: [{quat_odom.x:.4f}, {quat_odom.y:.4f}, {quat_odom.z:.4f}, {quat_odom.w:.4f}]")
            rospy.loginfo(f"  Euler ZYX (degrees): [{euler_odom[0]:.2f}, {euler_odom[1]:.2f}, {euler_odom[2]:.2f}]")
            if euler_cam_to_odom is not None:
                rospy.loginfo("CAMERA-TO-ODOM TRANSFORM:")
                rospy.loginfo(f"  Translation: [{trans_cam_to_odom[0]:.4f}, {trans_cam_to_odom[1]:.4f}, {trans_cam_to_odom[2]:.4f}]")
                rospy.loginfo(f"  Rotation Euler ZYX (degrees): [{euler_cam_to_odom[0]:.2f}, {euler_cam_to_odom[1]:.2f}, {euler_cam_to_odom[2]:.2f}]")
                rospy.loginfo(f"  Z-axis rotation difference: {euler_odom[0] - euler_cam[0]:.2f} degrees")
                rospy.loginfo(f"  Y-axis rotation difference: {euler_odom[1] - euler_cam[1]:.2f} degrees")
                rospy.loginfo(f"  X-axis rotation difference: {euler_odom[2] - euler_cam[2]:.2f} degrees")
            rospy.loginfo("="*60)
            rospy.loginfo("NOTE: The orientation difference is due to coordinate frame conventions:")
            rospy.loginfo("  - Camera frame: OpenCV convention (X right, Y down, Z forward)")
            rospy.loginfo("  - Odom frame: ROS convention (X forward, Y left, Z up)")
            rospy.loginfo("  - The camera-to-odom transform includes the camera's orientation on the robot")
            rospy.loginfo("="*60)
            self._rviz_orientation_debug_logged = True
        
        # Create mesh marker in odom frame
        # The pose position is at the object's center, but the mesh file has its origin at the bottom.
        # RViz places the mesh's origin at the marker position, so we need to offset the marker
        # position down by the offset from center to origin (in the object frame, then transform to odom frame).
        import numpy as np
        from scipy.spatial.transform import Rotation
        
        # Get object rotation in odom frame (will be reused for axes markers)
        obj_quat = pose_odom.pose.orientation
        obj_rot = Rotation.from_quat([obj_quat.x, obj_quat.y, obj_quat.z, obj_quat.w])
        obj_R = obj_rot.as_matrix()
        
        # Calculate offset from mesh origin to center in original mesh frame
        # inv(to_origin)[:3, 3] is the translation from origin to center
        inv_to_origin = np.linalg.inv(self.to_origin)
        mesh_origin_to_center = inv_to_origin[:3, 3]  # In original mesh frame
        
        # Transform offset to odom frame
        mesh_origin_to_center_odom = obj_R @ mesh_origin_to_center
        
        # Offset marker position: move DOWN by the offset (since mesh origin is below center)
        # This ensures the mesh's origin (bottom) is placed at the marker position, so the mesh center aligns with the pose center
        marker_position = np.array([
            pose_odom.pose.position.x,
            pose_odom.pose.position.y,
            pose_odom.pose.position.z
        ]) - mesh_origin_to_center_odom
        
        marker = Marker()
        marker.header = pose_odom.header
        marker.header.frame_id = pose_odom.header.frame_id  # Use odom frame
        marker.ns = "foundationpose"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose.position.x = marker_position[0]
        marker.pose.position.y = marker_position[1]
        marker.pose.position.z = marker_position[2]
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
        
        # Get object position in odom frame (obj_R already computed above for mesh marker)
        obj_pos = np.array([
            pose_odom.pose.position.x,
            pose_odom.pose.position.y,
            pose_odom.pose.position.z
        ])
        
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
