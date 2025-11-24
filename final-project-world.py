#!/usr/bin/env python3
"""
Final Project World 2 - Isaac Sim World Setup

This script creates an Isaac Sim world with:
- HSR robot positioned to face a table
- Coffee table with a mustard bottle on top
- Ground truth pose publisher for pose estimation evaluation
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library
import argparse
import ast
import os
import sys

# Third-party
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from rospkg import RosPack

# ROS package paths
rp = RosPack()
usd_repo_path = rp.get_path('usd')
template_repo_path = rp.get_path('hsr-omniverse')
template_world_repo_path = rp.get_path('rc_isaac_worlds')

# Add paths to sys.path (order matters - rc_isaac_worlds first for robocanes_isaac_world)
sys.path.append(template_world_repo_path)
sys.path.append(template_repo_path)

# Import robocanes_isaac_world FIRST to initialize Isaac Sim before robocanes_hsr is imported
# This is needed because robocanes_hsr imports omni.ui which requires Isaac Sim to be initialized
from robocanes_isaac_world import (
    robocanes_isaac_world, sim_app, Usd, Gf, omni, prims, 
    euler_angles_to_quat, RigidPrim
)

# Import pxr (USD library) AFTER Isaac Sim is initialized
# pxr is part of Isaac Sim and requires sim_app to be created first
# Note: Gf is already imported from robocanes_isaac_world, so we only need UsdGeom
from pxr import UsdGeom

# Import physx_utils AFTER Isaac Sim is initialized (sim_app must be created first)
# This import must happen after robocanes_isaac_world which creates sim_app
# Note: omni.physx is available after sim_app is created in robocanes_isaac_world
# We import it here, but if it fails, we'll import it lazily in the methods that use it
try:
    from omni.physx.scripts import utils as physx_utils
except (ImportError, ModuleNotFoundError):
    # If import fails, we'll import it lazily in the methods that need it
    physx_utils = None

# Now import robocanes_hsr (Isaac Sim is already initialized, so omni.ui will work)
import robocanes_hsr

# HSR-related utilities
from isaac_robot_behavior_start import isaac_robot_behavior_start
from isaac_robot_pose_pub import isaac_robot_pose_pub

# ============================================================================
# CONFIGURATION
# ============================================================================

# Parse command-line arguments for robot spawn position/orientation
parser = argparse.ArgumentParser(description="Final Project World Setup")
parser.add_argument(
    "--robot_spawn_pos_xyz", 
    type=str, 
    default='[3, 3.5, 0.01]', 
    help="Robot spawn position as xyz list."
)
parser.add_argument(
    "--robot_spawn_orient_xyz", 
    type=str, 
    default='[0, 0, 0]', 
    help="Robot spawn orientation as xyz list (will be overridden to face table)."
)
args, unknown_args = parser.parse_known_args()

# World configuration constants
TABLE_POSITION = [3, 4]  # Grid coordinates (x, y)
TABLE_HEIGHT = 0.1  # Table base height in meters
MUSTARD_HEIGHT_ABOVE_TABLE = 0.5  # Mustard bottle height above table in meters (when upright) - reduced for stability
MUSTARD_LYING_HEIGHT = 0.05  # Height when lying flat (half the width/diameter)
HEAD_TILT_ANGLE = -0.8  # Head tilt angle in radians (negative = down)
ARM_ROLL_JOINT_ANGLE = -1.57  # Arm roll joint angle in radians
WRIST_FLEX_JOINT_ANGLE = -1.57  # Wrist flex joint angle in radians

# ============================================================================
# MAIN WORLD CLASS
# ============================================================================

class final_project_world_2(robocanes_isaac_world):
    """
    Final Project World 2
    
    Creates a world with HSR robot, table, and mustard bottle.
    Robot is automatically oriented to face the table.
    """
    
    def __init__(self):
        """Initialize the world and add all objects."""
        print('[WORLD] final_project_world_2.__init__() called')
        super().__init__()
        print('[WORLD] super().__init__() completed')
        
        # Initialize pending flags for robot control
        self._pending_head_tilt = None
        self._pending_arm_neutral = None
        
        # Setup ROS publisher for ground truth pose
        print('[WORLD] Setting up ROS publisher...')
        self._setup_ros_publisher()
        print('[WORLD] ROS publisher setup completed')
        
        # Calculate robot orientation to face the table
        print('[WORLD] Calculating robot orientation...')
        robot_spawn_position = ast.literal_eval(args.robot_spawn_pos_xyz)
        # Store robot spawn position for later use (e.g., odom frame calculations)
        self.robot_spawn_position = robot_spawn_position
        print(f'[WORLD] Robot spawn position parsed: {robot_spawn_position}')
        robot_spawn_orientation = self._calculate_robot_orientation_to_table(
            robot_spawn_position, TABLE_POSITION
        )
        print(f'[WORLD] Robot spawn orientation calculated: {robot_spawn_orientation}')
        
        print(f'[WORLD] Robot spawn position: {robot_spawn_position}')
        print(f'[WORLD] Robot spawn orientation (facing table): {robot_spawn_orientation}')
        print(f'[WORLD]   -> yaw angle: {robot_spawn_orientation[2]:.3f} rad ({np.degrees(robot_spawn_orientation[2]):.1f} deg)')
        
        # Add objects to world
        print('[WORLD] Adding HSR robot...')
        self.add_hsr(
            robot_name='hsrb', 
            robot_spawn_position=robot_spawn_position, 
            robot_spawn_orientation=robot_spawn_orientation
        )
        print('[WORLD] HSR robot added')
        
        print('[WORLD] Adding table...')
        self.add_table_at_grid(TABLE_POSITION[0], TABLE_POSITION[1])
        print('[WORLD] Table added')
        
        print('[WORLD] Adding mustard bottle...')
        self.add_mustard_on_table(TABLE_POSITION[0], TABLE_POSITION[1])
        print('[WORLD] Mustard bottle added')
        
        print('[WORLD] __init__() completed successfully')
    
    # ========================================================================
    # ROS Setup
    # ========================================================================
    
    def _setup_ros_publisher(self):
        """Setup ROS publisher for ground truth mustard bottle pose."""
        print(f"[WORLD] _setup_ros_publisher() called")
        
        # Check if ROS is already initialized (e.g., by SemuRosBridge)
        print(f"[WORLD] Checking ROS node URI...")
        try:
            node_uri = rospy.get_node_uri()
            print(f"[WORLD] rospy.get_node_uri() returned: {node_uri}")
            if not node_uri:
                print(f"[WORLD] Initializing ROS node...")
                rospy.init_node('final_project_world', anonymous=True, disable_signals=True)
                print(f"[WORLD] ROS node initialized")
            else:
                print(f"[WORLD] ROS node already initialized")
        except Exception as e:
            print(f"[WORLD] ERROR in ROS node check/init: {e}")
            import traceback
            print(traceback.format_exc())
        
        print(f"[WORLD] Creating Publisher object...")
        try:
            self.mustard_gt_pub = rospy.Publisher(
                '/mustard_bottle/ground_truth_pose', 
                PoseStamped, 
                queue_size=10,
                latch=True  # Latch the last message so subscribers get it immediately
            )
            print(f"[WORLD] Created ground truth pose publisher on /mustard_bottle/ground_truth_pose")
        except Exception as e:
            print(f"[WORLD] ERROR creating Publisher: {e}")
            import traceback
            print(traceback.format_exc())
            raise
        
        print(f"[WORLD] _setup_ros_publisher() returning")
    
    # ========================================================================
    # Robot Setup
    # ========================================================================
    
    def _calculate_robot_orientation_to_table(self, robot_pos, table_pos):
        """
        Calculate robot orientation to face the table.
        
        Args:
            robot_pos: Robot position [x, y, z]
            table_pos: Table position [x, y]
            
        Returns:
            List of Euler angles [roll, pitch, yaw] in radians
        """
        dx = table_pos[0] - robot_pos[0]
        dy = table_pos[1] - robot_pos[1]
        yaw_angle = np.arctan2(dy, dx)
        return [0.0, 0.0, yaw_angle]
    
    def add_hsr(self, robot_name, robot_spawn_position, robot_spawn_orientation):
        """
        Add HSR robot to the world.
        
        Args:
            robot_name: Name of the robot (e.g., 'hsrb')
            robot_spawn_position: Robot spawn position [x, y, z]
            robot_spawn_orientation: Robot spawn orientation [roll, pitch, yaw] in radians
        """
        self.hsr_instance = robocanes_hsr.hsr(
            prefix=f'/{robot_name}',
            spawn_config={
                'translation': robot_spawn_position,
                'orientation': robot_spawn_orientation,
                'scale': [1, 1, 1]
            }
        )
        
        self.hsr_instance.onsimulationstart(self.sim_world)
        
        # Setup ROS publishers for robot pose
        self.isaac_robot_behavior_start = isaac_robot_behavior_start()
        self.isaac_robot_pose_pub = isaac_robot_pose_pub()
        
        # Set initial robot pose
        self.set_head_tilt_position(HEAD_TILT_ANGLE)
        self.set_arm_neutral_position()
    
    def set_head_tilt_position(self, tilt_angle):
        """
        Set the head tilt joint position to look down at the table.
        
        Args:
            tilt_angle (float): Head tilt angle in radians. Negative values tilt down.
                                 Range: -1.57 to 0.52 radians (from URDF limits)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from omni.isaac.dynamic_control import _dynamic_control
            dc = _dynamic_control.acquire_dynamic_control_interface()
            art = dc.get_articulation('/World/hsrb')
            
            if art != _dynamic_control.INVALID_HANDLE:
                head_tilt_dof = dc.find_articulation_dof(art, "head_tilt_joint")
                if head_tilt_dof != _dynamic_control.DofType.DOF_NONE:
                    dc.set_dof_position_target(head_tilt_dof, tilt_angle)
                    print(f"[WORLD] Set head_tilt_joint to {tilt_angle:.3f} rad ({tilt_angle * 180 / np.pi:.1f} deg)")
                    self._pending_head_tilt = None
                    return True
                else:
                    print(f"[WORLD] Warning: Could not find head_tilt_joint DOF")
            else:
                print(f"[WORLD] Warning: Articulation not yet available, head tilt will be set after simulation starts")
                self._pending_head_tilt = tilt_angle
                return False
        except Exception as e:
            print(f"[WORLD] Warning: Could not set head tilt position: {e}")
            self._pending_head_tilt = tilt_angle
            return False
    
    def set_arm_neutral_position(self):
        """
        Set the arm to neutral position using MoveGroupCommander or dynamic_control.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Try MoveGroupCommander first (preferred, works with ROS)
        try:
            import rospy
            import moveit_commander
            
            if not rospy.get_node_uri():
                print(f"[WORLD] Warning: ROS node not initialized, arm will be set after simulation starts")
                self._pending_arm_neutral = True
                return False
            
            move_group = moveit_commander.MoveGroupCommander("arm")
            joint_values = move_group.get_current_joint_values()
            active_joints = move_group.get_active_joints()
            
            # Set both joints at once
            if "arm_roll_joint" in active_joints:
                joint_values[active_joints.index("arm_roll_joint")] = ARM_ROLL_JOINT_ANGLE
            if "wrist_flex_joint" in active_joints:
                joint_values[active_joints.index("wrist_flex_joint")] = WRIST_FLEX_JOINT_ANGLE
            
            move_group.go(joint_values, wait=True)
            move_group.stop()
            
            print(f"[WORLD] Set arm to neutral position using MoveGroupCommander")
            self._pending_arm_neutral = None
            return True
            
        except Exception as e:
            print(f"[WORLD] MoveGroupCommander failed: {e}, trying dynamic_control")
            # Fallback to dynamic_control
            return self._set_arm_neutral_dynamic_control()
    
    def _set_arm_neutral_dynamic_control(self):
        """Set arm to neutral position using dynamic_control (fallback method)."""
        try:
            from omni.isaac.dynamic_control import _dynamic_control
            dc = _dynamic_control.acquire_dynamic_control_interface()
            art = dc.get_articulation('/World/hsrb')
            
            if art == _dynamic_control.INVALID_HANDLE:
                print(f"[WORLD] Warning: Articulation not yet available, arm will be set after simulation starts")
                self._pending_arm_neutral = True
                return False
            
            arm_roll_dof = dc.find_articulation_dof(art, "arm_roll_joint")
            if arm_roll_dof != _dynamic_control.DofType.DOF_NONE:
                dc.set_dof_position_target(arm_roll_dof, ARM_ROLL_JOINT_ANGLE)
            
            wrist_flex_dof = dc.find_articulation_dof(art, "wrist_flex_joint")
            if wrist_flex_dof != _dynamic_control.DofType.DOF_NONE:
                dc.set_dof_position_target(wrist_flex_dof, WRIST_FLEX_JOINT_ANGLE)
            
            print(f"[WORLD] Set arm to neutral position using dynamic_control")
            self._pending_arm_neutral = None
            return True
            
        except Exception as dc_e:
            print(f"[WORLD] Warning: Could not set arm neutral position: {dc_e}")
            self._pending_arm_neutral = True
            return False
    
    # ========================================================================
    # Object Placement
    # ========================================================================
    
    def add_table_at_grid(self, grid_x, grid_y):
        """
        Add a coffee table at the specified grid coordinates.
        Table is oriented horizontally (aligned to grid axes) with front facing the robot.
        
        Args:
            grid_x: Grid X coordinate (in meters)
            grid_y: Grid Y coordinate (in meters)
        """
        table_usd = os.path.join(usd_repo_path, 'robocanes_lab', 'robocanes_lab', 'coffeeTable.usd')
        
        # Set table to horizontal orientation (aligned to grid axes)
        # The coffeeTable.usd model appears to have a default 45-degree rotation
        # Compensate by rotating -45 degrees to align table edges with X/Y axes
        # This ensures the table is horizontal (not diagonal) for easier robot approach
        table_yaw = 0#-np.pi / 4  # -45 degrees to compensate for model's default rotation
        
        table_prim = prims.create_prim(
            prim_path=f'/World/Props/coffee_table',
            usd_path=table_usd,
            translation=[grid_x, grid_y, TABLE_HEIGHT],
            orientation=euler_angles_to_quat([0.0, 0.0, table_yaw]),  # Horizontal orientation
            scale=[1, 1, 1],
            semantic_label='coffee_table'
        )
        
        # Enable collision for the table so objects can rest on it
        global physx_utils
        if physx_utils is None:
            from omni.physx.scripts import utils as physx_utils
        physx_utils.setCollider(prim=table_prim, approximationShape='sdfMesh')
        
        # Make table kinematic (non-movable) so it doesn't rotate or move
        # This prevents the table from rotating after mustard is placed
        try:
            from pxr import UsdPhysics
            # Apply RigidBodyAPI and set as kinematic
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(table_prim)
            if rigid_body_api:
                # Set as kinematic so it doesn't move/rotate
                rigid_body_api.CreateKinematicEnabledAttr(True)
                print(f'[WORLD] Set table as kinematic (non-movable)')
        except Exception as e:
            print(f'[WORLD] Warning: Could not set table as kinematic: {e}')
        
        # Store table prim reference to ensure orientation doesn't change
        self.table_prim = table_prim
        
        print(f'[WORLD] Added table at grid location ({grid_x}, {grid_y}) with horizontal orientation yaw={table_yaw:.3f} rad ({np.degrees(table_yaw):.1f} deg)')
    
    def add_mustard_on_table(self, grid_x, grid_y):
        """
        Add mustard bottle on top of the table.
        
        Args:
            grid_x: Grid X coordinate (same as table)
            grid_y: Grid Y coordinate (same as table)
        """
        # Calculate mustard position (on top of table)
        mustard_pos = [grid_x, grid_y, TABLE_HEIGHT + MUSTARD_HEIGHT_ABOVE_TABLE]
        
        # YCB object path
        mustard_usd = os.path.join(
            usd_repo_path, 'ycb', '006_mustard_bottle', 
            'google_16k_converted', 'textured_obj.usd'
        )
        
        if not os.path.exists(mustard_usd):
            print(f'[WORLD] Warning: Mustard USD not found at {mustard_usd}')
            return
        
        # Create mustard prim with default orientation
        mustard_prim = prims.create_prim(
            prim_path='/World/Props/mustard_bottle',
            usd_path=mustard_usd,
            translation=mustard_pos,
            orientation=euler_angles_to_quat([0.0, 0.0, 0.0]),
            scale=[1, 1, 1],
            semantic_label='006_mustard_bottle'
        )
        
        # Enable collision and physics
        global physx_utils
        if physx_utils is None:
            from omni.physx.scripts import utils as physx_utils
        physx_utils.setCollider(prim=mustard_prim, approximationShape='sdfMesh')
        
        # Create rigid body for mustard bottle
        RigidPrim(prim_path=str(mustard_prim.GetPrimPath()), name='mustard_bottle')
        
        # Set mustard bottle mass to prevent it from falling through table
        # Also ensure it's not kinematic so it can be picked up later
        try:
            from pxr import UsdPhysics
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(mustard_prim)
            if rigid_body_api:
                # Set reasonable mass (in kg) - not too light, not too heavy
                mass_api = UsdPhysics.MassAPI.Apply(mustard_prim)
                if mass_api:
                    mass_api.CreateMassAttr(0.5)  # 0.5 kg - reasonable for a mustard bottle
                    print(f'[WORLD] Set mustard bottle mass to 0.5 kg')
        except Exception as e:
            print(f'[WORLD] Warning: Could not set mustard bottle mass: {e}')
        
        # Store mustard prim for later pose publishing
        self.mustard_prim = mustard_prim
        
        print(f'[WORLD] Added mustard bottle on table at ({grid_x}, {grid_y}) with default orientation')
    
    # ========================================================================
    # Ground Truth Pose Publishing
    # ========================================================================
    
    def publish_ground_truth_pose(self):
        """
        Publish ground truth mustard bottle pose in camera frame.
        This is used for comparing with FoundationPose estimates.
        """
        # Track call count (no logging to reduce spam)
        if not hasattr(self, '_gt_call_count'):
            self._gt_call_count = 0
        self._gt_call_count += 1
        
        # Debug: Print on first call
        if self._gt_call_count == 1:
            print(f"[WORLD] DEBUG - publish_ground_truth_pose called for the first time", flush=True)
            print(f"[WORLD] DEBUG - Checking mustard_prim...", flush=True)
        
        # Check mustard_prim
        if not hasattr(self, 'mustard_prim'):
            if not hasattr(self, '_gt_no_mustard_attr_logged'):
                print(f"[WORLD] ERROR: mustard_prim attribute does not exist!")
                print(f"[WORLD] ERROR: mustard_prim attribute does not exist!", flush=True)
                self._gt_no_mustard_attr_logged = True
            return
        
        if not self.mustard_prim:
            if not hasattr(self, '_gt_no_mustard_value_logged'):
                print(f"[WORLD] ERROR: mustard_prim is None or invalid!")
                print(f"[WORLD] ERROR: mustard_prim is None or invalid!", flush=True)
                self._gt_no_mustard_value_logged = True
            return
        
        if self._gt_call_count == 1:
            print(f"[WORLD] DEBUG - mustard_prim exists and is valid", flush=True)
        
        try:
            # Debug: Check if publisher is ready
            if not hasattr(self, '_gt_pub_ready_logged'):
                num_subscribers = self.mustard_gt_pub.get_num_connections()
                print(f"[WORLD] Ground truth publisher has {num_subscribers} subscriber(s)")
                self._gt_pub_ready_logged = True
            
            # Get camera world pose - try multiple possible paths
            # Use head_rgbd_sensor_link as it's the parent of head_rgbd_sensor_rgb_frame
            # The ROS frame head_rgbd_sensor_rgb_frame is typically a child of head_rgbd_sensor_link
            camera_prim_paths = [
                '/World/hsrb/head_rgbd_sensor_link',  # Use link, not Camera prim
                '/World/hsrb/head_rgbd_sensor_link/Camera',  # Fallback to Camera prim
                '/World/hsrb/head_rgbd_sensor_rgb_frame',
                '/World/hsrb/head_rgbd_sensor/rgb_frame',
                '/World/hsrb/head_rgbd_sensor',
            ]
            camera_prim = None
            camera_prim_path = None
            for path in camera_prim_paths:
                prim = self.stage.GetPrimAtPath(path)
                if prim and prim.IsValid():
                    # Check if it's actually a camera or xformable
                    if prim.IsA(UsdGeom.Camera) or prim.IsA(UsdGeom.Xformable):
                        camera_prim = prim
                        camera_prim_path = path
                        if not hasattr(self, '_gt_camera_path_logged'):
                            print(f"[WORLD] Found camera at: {path}")
                            self._gt_camera_path_logged = True
                        break
            
            if not camera_prim or not camera_prim_path:
                if not hasattr(self, '_gt_no_camera_logged'):
                    print(f"[WORLD] ERROR: Camera prim not found! Tried paths: {camera_prim_paths}")
                    # List available prims under /World/hsrb for debugging
                    try:
                        hsrb_prim = self.stage.GetPrimAtPath('/World/hsrb')
                        if hsrb_prim and hsrb_prim.IsValid():
                            print(f"[WORLD] Available prims under /World/hsrb:")
                            def list_children(prim, depth=0, max_depth=3):
                                if depth > max_depth:
                                    return
                                indent = "  " * (depth + 1)
                                print(f"{indent}- {prim.GetPath()}")
                                for child in prim.GetChildren():
                                    list_children(child, depth + 1, max_depth)
                            list_children(hsrb_prim)
                    except Exception as e:
                        print(f"[WORLD] Error listing prims: {e}")
                    self._gt_no_camera_logged = True
                return
            
            # Get mustard bottle pose in world frame (simple - let the node handle transformations)
            mustard_world_transform = omni.usd.get_world_transform_matrix(self.mustard_prim)
            
            # Extract position and rotation in world frame
            mustard_world_pos = np.array(mustard_world_transform.ExtractTranslation())
            mustard_world_rot_quat = mustard_world_transform.ExtractRotationQuat()
            mustard_world_rot = Gf.Quatd(
                mustard_world_rot_quat.real,
                mustard_world_rot_quat.imaginary[0],
                mustard_world_rot_quat.imaginary[1],
                mustard_world_rot_quat.imaginary[2]
            )
            
            # Convert world position to grid coordinates (grid is 1:1 with world coordinates in meters)
            mustard_grid_x = mustard_world_pos[0]
            mustard_grid_y = mustard_world_pos[1]
            
            
            # Debug: Print world frame pose (simple - transformations handled by node)
            if not hasattr(self, '_gt_debug_logged'):
                print(f"[WORLD] Publishing ground truth pose in world/map frame:", flush=True)
                print(f"  World position:    ({mustard_world_pos[0]:.4f}, {mustard_world_pos[1]:.4f}, {mustard_world_pos[2]:.4f}) m", flush=True)
                print(f"  World orientation: ({mustard_world_rot.imaginary[0]:.4f}, {mustard_world_rot.imaginary[1]:.4f}, {mustard_world_rot.imaginary[2]:.4f}, {mustard_world_rot.real:.4f})", flush=True)
                print(f"  Grid coordinates:  ({mustard_grid_x:.2f}, {mustard_grid_y:.2f})", flush=True)
                print(f"[WORLD] Node will handle transformation to camera frame", flush=True)
                self._gt_debug_logged = True
            
            # Create and publish PoseStamped message in world frame
            # The node will handle transformation to camera frame
            gt_pose = PoseStamped()
            gt_pose.header.frame_id = 'map'  # Use 'map' frame (or 'odom' if map not available)
            gt_pose.header.stamp = rospy.Time.now()
            # Publish in world frame - node will transform
            gt_pose.pose.position.x = float(mustard_world_pos[0])
            gt_pose.pose.position.y = float(mustard_world_pos[1])
            gt_pose.pose.position.z = float(mustard_world_pos[2])
            # Gf.Quatd: (real, imag_x, imag_y, imag_z) = (w, x, y, z)
            gt_pose.pose.orientation.x = float(mustard_world_rot.imaginary[0])
            gt_pose.pose.orientation.y = float(mustard_world_rot.imaginary[1])
            gt_pose.pose.orientation.z = float(mustard_world_rot.imaginary[2])
            gt_pose.pose.orientation.w = float(mustard_world_rot.real)
            
            # Publish the pose in world/map frame
            # The node will handle transformation to camera frame for comparison
            self.mustard_gt_pub.publish(gt_pose)
            
            # Debug: Log first successful publish only
            if not hasattr(self, '_gt_first_publish_logged'):
                print(f"[WORLD] Published first ground truth pose (in {gt_pose.header.frame_id} frame): pos=[{gt_pose.pose.position.x:.3f}, {gt_pose.pose.position.y:.3f}, {gt_pose.pose.position.z:.3f}], "
                      f"orient=[{gt_pose.pose.orientation.x:.3f}, {gt_pose.pose.orientation.y:.3f}, {gt_pose.pose.orientation.z:.3f}, {gt_pose.pose.orientation.w:.3f}]", flush=True)
                self._gt_first_publish_logged = True
            
        except Exception as e:
            # Log error instead of silently failing - ALWAYS print errors
            print(f"[WORLD] ERROR in publish_ground_truth_pose: {e}", flush=True)
            import traceback
            print(traceback.format_exc(), flush=True)
            if not hasattr(self, '_gt_error_logged'):
                self._gt_error_logged = True
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def decompose_matrix(self, mat: Gf.Matrix4d):
        """
        Decompose a 4x4 transformation matrix into translation, rotation, and scale.
        
        Reference: https://forums.developer.nvidia.com/t/get-euler-angles-rotation-of-a-prim/275600
        
        Args:
            mat: Gf.Matrix4d transformation matrix
            
        Returns:
            Tuple of (translation, rotation, scale) as numpy arrays
        """
        reversed_ident_mtx = reversed(Gf.Matrix3d())
        translate = Gf.Vec3d(mat.ExtractTranslation())
        scale = Gf.Vec3d(*(v.GetLength() for v in mat.ExtractRotationMatrix()))
        
        mat.Orthonormalize()
        rotate = Gf.Vec3d(*reversed(mat.ExtractRotation().Decompose(*reversed_ident_mtx)))
        
        return np.array(translate), np.array(rotate), np.array(scale)
    
    def get_world_transform_xform(self, prim: Usd.Prim):
        """
        Get world transform of a prim decomposed into translation, rotation, and scale.
        
        Args:
            prim: USD prim to get transform for
            
        Returns:
            Tuple of (translation, rotation, scale) as numpy arrays
        """
        world_transform = omni.usd.get_world_transform_matrix(prim)
        return self.decompose_matrix(world_transform)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Debug: Confirm script execution started
    print('[WORLD] ========================================')
    print('[WORLD] Script execution started')
    print('[WORLD] ========================================')
    
    # Create world instance
    print('[WORLD] Creating world instance...')
    try:
        world = final_project_world_2()
        print('[WORLD] World instance created successfully')
    except Exception as e:
        print(f'[WORLD] ERROR creating world: {e}')
        import traceback
        print(traceback.format_exc())
        raise
    
    # Simulation state tracking
    sim_world_onetime_trigger = True
    head_tilt_set = False
    arm_neutral_set = False
    
    # Debug: Confirm main loop starting
    print('[WORLD] ========================================')
    print('[WORLD] *** Main simulation loop starting ***')
    print('[WORLD] ========================================')
    loop_count = 0
    
    # Main simulation loop
    while sim_app.is_running():
        loop_count += 1
        # Only log first iteration to reduce spam
        if loop_count == 1:
            print(f'[WORLD] Main loop iteration {loop_count}, is_playing={world.sim_world.is_playing()}')
        
        # Step simulation
        world.sim_world.step(render=True)
        world.hsr_instance.step()
        world.sim_world.play()
        
        # Handle one-time setup when simulation starts playing
        if world.sim_world.is_playing():
            if sim_world_onetime_trigger:
                print('[WORLD] ISAAC WORLD ONETIME PLAY!!!')
                print('[WORLD] ISAAC WORLD ONETIME PLAY!!!', flush=True)
                sim_world_onetime_trigger = False
                
                # Ensure table orientation is locked after simulation starts
                # This prevents any physics or other systems from rotating the table
                if hasattr(world, 'table_prim') and world.table_prim:
                    try:
                        # Re-apply kinematic setting to ensure table doesn't move
                        from pxr import UsdPhysics
                        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(world.table_prim)
                        if rigid_body_api:
                            rigid_body_api.CreateKinematicEnabledAttr(True)
                            print('[WORLD] Locked table orientation after simulation start')
                    except Exception as e:
                        print(f'[WORLD] Warning: Could not lock table orientation: {e}')
            
            # Set head tilt after simulation starts when articulation is available
            if not head_tilt_set and hasattr(world, '_pending_head_tilt') and world._pending_head_tilt is not None:
                if world.set_head_tilt_position(world._pending_head_tilt):
                    head_tilt_set = True
            
            # Set arm neutral position after simulation starts when articulation is available
            if not arm_neutral_set and hasattr(world, '_pending_arm_neutral') and world._pending_arm_neutral is not None:
                if world.set_arm_neutral_position():
                    arm_neutral_set = True
        
        # Start robot behavior
        world.isaac_robot_behavior_start.start()
        
        # Publish robot pose
        chosen_prim = world.stage.GetPrimAtPath('/World/hsrb/base_footprint')
        world_translate, world_rotate, world_scale = world.get_world_transform_xform(prim=chosen_prim)
        prim_world_pose = np.array([world_translate[0], world_translate[1], np.radians(world_rotate[2])])
        world.isaac_robot_pose_pub.publish(pose=prim_world_pose)
        
        # Publish ground truth mustard bottle pose
        # Only publish when simulation is playing (prims need to be ready)
        is_playing = world.sim_world.is_playing()
        if is_playing:
            try:
                world.publish_ground_truth_pose()
            except Exception as e:
                import sys
                import traceback
                sys.stdout.write(f'[WORLD] ERROR calling publish_ground_truth_pose: {e}\n')
                sys.stdout.write(traceback.format_exc())
                sys.stdout.flush()
        else:
            if loop_count <= 5 or loop_count % 100 == 0:
                import sys
                sys.stdout.write(f'[WORLD] DEBUG - Simulation NOT playing, loop={loop_count}, is_playing={is_playing}\n')
                sys.stdout.flush()
    
    # Cleanup
    sim_app.close()
