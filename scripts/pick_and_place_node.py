#!/usr/bin/env python3
"""
Simple Pick and Place ROS Node for HSR Robot

This node subscribes to object pose estimates from FoundationPose and executes
simple pick operations using MoveIt for motion planning.
"""

import rospy
import moveit_commander
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String, Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import numpy as np
import math

class PickAndPlaceNode:
    """Simple Pick and Place Node for HSR Robot."""
    
    STATE_IDLE = 'idle'
    STATE_PREPARING = 'preparing'
    STATE_APPROACHING = 'approaching'
    STATE_GRASPING = 'grasping'
    STATE_LIFTING = 'lifting'
    STATE_COMPLETE = 'complete'
    STATE_FAILED = 'failed'
    
    def __init__(self):
        """Initialize the pick and place node."""
        rospy.init_node('pick_and_place', anonymous=True)
        
        # Load parameters
        self.pose_topic = rospy.get_param('~pick/pose_topic', '/foundationpose_pose_estimation/pose')
        self.approach_height = rospy.get_param('~pick/approach_height', 0.15)
        self.grasp_height_offset = rospy.get_param('~pick/grasp_height_offset', 0.02)
        self.lift_height = rospy.get_param('~pick/lift_height', 0.20)
        self.gripper_open_position = rospy.get_param('~pick/gripper_open_position', 1.0)
        self.gripper_close_position = rospy.get_param('~pick/gripper_close_position', 0.0)
        self.end_effector_orientation = rospy.get_param('~pick/end_effector_orientation', [0.0, math.pi, 0.0])
        self.auto_pick = rospy.get_param('~pick/auto_pick', True)
        self.planning_timeout = rospy.get_param('~pick/planning_timeout', 20.0)
        self.test_joints_on_init = rospy.get_param('~pick/test_joints_on_init', True)
        
        # Object dimensions for calculating top surface position
        # Format: [width, depth, height] in meters (X, Y, Z in object frame)
        self.object_dimensions = rospy.get_param('~pick/object_dimensions', [0.164, 0.213, 0.072])
        self.object_height = self.object_dimensions[2]  # Z dimension is height
        rospy.loginfo(f">>> [PICK] Object dimensions: {self.object_dimensions} m (width x depth x height)")
        rospy.loginfo(f">>> [PICK] Object height: {self.object_height:.3f} m - will use this to calculate top surface")
        
        # Initialize MoveIt
        moveit_commander.roscpp_initialize([])
        self.arm = moveit_commander.MoveGroupCommander('arm')
        self.gripper = moveit_commander.MoveGroupCommander('gripper')
        self.arm.set_max_velocity_scaling_factor(0.3)
        self.arm.set_max_acceleration_scaling_factor(0.3)
        self.arm.set_planning_time(self.planning_timeout)
        self.arm.set_num_planning_attempts(10)  # Try multiple times
        
        # Initialize robot state for IK checking
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        
        # Set planner - RRTConnect is usually fastest and most reliable
        # Other options: 'RRT', 'PRM', 'RRTstar', 'TRRT', 'EST', 'SBL'
        self.arm.set_planner_id('RRTConnect')
        rospy.loginfo(f">>> [MOVEIT] Using planner: RRTConnect")
        
        # Allow replanning if needed
        self.arm.allow_replanning(True)
        
        # Initialize TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Wait for MoveIt to synchronize with current robot state
        rospy.loginfo(">>> [MOVEIT] Waiting for robot state synchronization...")
        rospy.sleep(2.0)  # Give time for joint_states to be received
        
        # Verify we can get current state
        try:
            current_joint_values = self.arm.get_current_joint_values()
            joint_names = self.arm.get_active_joints()
            current_pose = self.arm.get_current_pose().pose
            rospy.loginfo(f">>> [MOVEIT] Robot state synchronized. Active joints: {joint_names}")
            rospy.loginfo(f">>> [MOVEIT] Current joint values: {dict(zip(joint_names, [f'{v:.3f}' for v in current_joint_values]))}")
            rospy.loginfo(f">>> [MOVEIT] Current end-effector pose: ({current_pose.position.x:.3f}, {current_pose.position.y:.3f}, {current_pose.position.z:.3f})")
            
            # Check if arm_lift_joint is in the active joints
            if 'arm_lift_joint' in joint_names:
                lift_idx = joint_names.index('arm_lift_joint')
                rospy.loginfo(f">>> [MOVEIT] arm_lift_joint value: {current_joint_values[lift_idx]:.3f} m (prismatic joint for vertical lift)")
            else:
                rospy.logwarn(">>> [MOVEIT] WARNING: arm_lift_joint not found in active joints!")
        except Exception as e:
            rospy.logwarn(f">>> [MOVEIT] Could not get initial robot state: {e}")
            rospy.logwarn(">>> [MOVEIT] This is OK if robot hasn't started publishing joint_states yet")
        
        # TEST: Verify all joints can move by testing each one (optional)
        if self.test_joints_on_init:
            rospy.loginfo("="*70)
            rospy.loginfo(">>> [INIT_TEST] Testing joint movement capability...")
            rospy.loginfo("="*70)
            self._test_joint_movement()
            rospy.loginfo("="*70)
            rospy.loginfo(">>> [INIT_TEST] Joint movement test complete")
            rospy.loginfo("="*70)
        else:
            rospy.loginfo(">>> [INIT_TEST] Joint movement test disabled (set ~pick/test_joints_on_init to True to enable)")
        
        # Setup ROS communication
        self.pose_sub = rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback, queue_size=1)
        self.status_pub = rospy.Publisher('~status', String, queue_size=10, latch=True)
        self.pick_complete_pub = rospy.Publisher('~pick_complete', Bool, queue_size=10, latch=True)
        
        # Visualization markers
        self.target_pose_marker_pub = rospy.Publisher('~target_pose_marker', Marker, queue_size=10, latch=True)
        self.end_effector_marker_pub = rospy.Publisher('~end_effector_marker', Marker, queue_size=10, latch=True)
        self.path_marker_pub = rospy.Publisher('~path_marker', Marker, queue_size=10, latch=True)
        self.marker_id_counter = 0
        
        if not self.auto_pick:
            self.trigger_sub = rospy.Subscriber('~trigger_pick', Bool, self.trigger_pick_callback, queue_size=1)
        
        # State machine
        self.current_state = self.STATE_IDLE
        self.target_pose = None
        self.pick_triggered = False
        
        # Pose buffer for consistency checking
        self.pose_buffer = []
        self.pose_buffer_size = rospy.get_param('~pick/pose_buffer_size', 1)
        self.pose_consistency_threshold = rospy.get_param('~pick/pose_consistency_threshold', 0.05)  # 5cm position threshold
        self.pose_buffer_consistent = False  # Flag to stop processing new poses once consistent
        
        rospy.loginfo("Pick and Place node initialized")
        rospy.loginfo(f"Subscribing to: {self.pose_topic}")
        rospy.loginfo(f"Auto-pick: {self.auto_pick}")
        rospy.loginfo(f"Pose buffer size: {self.pose_buffer_size}, consistency threshold: {self.pose_consistency_threshold}m")
    
    def _test_joint_movement(self):
        """Test that all joints can actually move by moving each one slightly."""
        try:
            joint_names = self.arm.get_active_joints()
            current_joint_values = self.arm.get_current_joint_values()
            initial_values = dict(zip(joint_names, current_joint_values))
            
            rospy.loginfo(f">>> [INIT_TEST] Starting joint movement test with {len(joint_names)} joints")
            rospy.loginfo(f">>> [INIT_TEST] Initial joint values: {dict(zip(joint_names, [f'{v:.3f}' for v in current_joint_values]))}")
            
            # Test each joint with a small movement
            test_results = {}
            for i, joint_name in enumerate(joint_names):
                rospy.loginfo(f">>> [INIT_TEST] Testing joint {i+1}/{len(joint_names)}: {joint_name}")
                try:
                    initial_value = current_joint_values[i]
                    
                    # Calculate a small test movement (different for each joint type)
                    if joint_name == 'arm_lift_joint':
                        # Prismatic joint: move 0.05m
                        test_value = min(0.69, initial_value + 0.05)
                        if test_value == initial_value:
                            test_value = min(0.69, 0.1)  # If at 0, move to 0.1
                    elif 'flex' in joint_name or 'roll' in joint_name:
                        # Revolute joints: move 0.1 rad (~6 degrees)
                        test_value = initial_value + 0.1
                    else:
                        # Other joints: small movement
                        test_value = initial_value + 0.05
                    
                    # Set target for this joint only
                    joint_target = dict(zip(joint_names, current_joint_values))
                    joint_target[joint_name] = test_value
                    
                    rospy.loginfo(f">>> [INIT_TEST]   Moving {joint_name} from {initial_value:.3f} to {test_value:.3f}")
                    self.arm.set_joint_value_target(joint_target)
                    self.arm.set_planning_time(5.0)  # Short timeout for test
                    
                    # Try to move
                    success = self.arm.go(wait=True)
                    
                    if success:
                        # Check if it actually moved
                        new_values = self.arm.get_current_joint_values()
                        new_value = new_values[i]
                        moved = abs(new_value - initial_value) > 0.01  # At least 1cm or 0.01 rad
                        
                        if moved:
                            rospy.loginfo(f">>> [INIT_TEST]   ✓ {joint_name} MOVED: {initial_value:.3f} -> {new_value:.3f}")
                            test_results[joint_name] = True
                        else:
                            rospy.logwarn(f">>> [INIT_TEST]   ✗ {joint_name} did not move (stayed at {new_value:.3f})")
                            test_results[joint_name] = False
                        
                        # Update current values for next test
                        current_joint_values = new_values
                    else:
                        rospy.logwarn(f">>> [INIT_TEST]   ✗ {joint_name} movement FAILED (go() returned False)")
                        test_results[joint_name] = False
                    
                    rospy.sleep(0.5)  # Brief pause between tests
                    
                except Exception as e:
                    rospy.logerr(f">>> [INIT_TEST]   ✗ {joint_name} test ERROR: {e}")
                    test_results[joint_name] = False
                    import traceback
                    rospy.logerr(traceback.format_exc())
            
            # Summary
            rospy.loginfo(">>> [INIT_TEST] Test Summary:")
            all_passed = True
            for joint_name, passed in test_results.items():
                status = "✓ PASS" if passed else "✗ FAIL"
                rospy.loginfo(f">>> [INIT_TEST]   {joint_name}: {status}")
                if not passed:
                    all_passed = False
            
            if all_passed:
                rospy.loginfo(">>> [INIT_TEST] ✓ All joints can move!")
            else:
                rospy.logwarn(">>> [INIT_TEST] ✗ Some joints failed to move - this may cause planning issues")
            
            # Move back to initial position (or 'go' position)
            rospy.loginfo(">>> [INIT_TEST] Moving arm back to 'go' position...")
            try:
                self.arm.set_named_target('go')
                self.arm.set_planning_time(10.0)
                success = self.arm.go(wait=True)
                if success:
                    rospy.loginfo(">>> [INIT_TEST] ✓ Arm returned to 'go' position")
                else:
                    rospy.logwarn(">>> [INIT_TEST] ✗ Failed to return to 'go' position")
            except Exception as e:
                rospy.logwarn(f">>> [INIT_TEST] ✗ Error returning to 'go' position: {e}")
            
            rospy.sleep(1.0)  # Final pause
            
        except Exception as e:
            rospy.logerr(f">>> [INIT_TEST] Joint movement test failed: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def pose_callback(self, msg):
        """Callback for object pose updates."""
        # Stop processing new poses once buffer is consistent
        if self.pose_buffer_consistent:
            rospy.loginfo_throttle(5.0, ">>> [POSE_BUFFER] Already consistent, ignoring new poses (using locked pose)")
            return
        
        rospy.loginfo(f">>> [POSE_BUFFER] Received new pose: ({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})")
        
        # Add pose to buffer
        self.pose_buffer.append(msg)
        
        # Keep only last N poses
        if len(self.pose_buffer) > self.pose_buffer_size:
            removed_pose = self.pose_buffer.pop(0)
            rospy.loginfo(f">>> [POSE_BUFFER] Removed oldest pose from buffer (now {len(self.pose_buffer)}/{self.pose_buffer_size})")
        
        rospy.loginfo(f">>> [POSE_BUFFER] Buffer status: {len(self.pose_buffer)}/{self.pose_buffer_size} poses")
        
        # Check if we have enough poses and they're consistent
        if len(self.pose_buffer) >= self.pose_buffer_size:
            rospy.loginfo(f">>> [POSE_BUFFER] Checking consistency of {len(self.pose_buffer)} poses...")
            if self._check_pose_consistency():
                # Mark buffer as consistent and stop processing new poses
                self.pose_buffer_consistent = True
                
                # Use the average of consistent poses
                self.target_pose = self._get_average_pose()
                rospy.loginfo("="*60)
                rospy.loginfo(">>> [POSE_BUFFER] *** POSE BUFFER CONSISTENT ***")
                rospy.loginfo(f">>> [POSE_BUFFER] Using average pose: ({self.target_pose.pose.position.x:.3f}, "
                            f"{self.target_pose.pose.position.y:.3f}, {self.target_pose.pose.position.z:.3f})")
                rospy.loginfo(">>> [POSE_BUFFER] STOPPING pose buffer updates - locked in this pose")
                rospy.loginfo("="*60)
                
                # Publish target pose marker
                self._publish_target_pose_marker(self.target_pose, color=(1.0, 0.0, 0.0, 1.0), scale=0.15)
                
                if self.auto_pick and self.current_state == self.STATE_IDLE:
                    rospy.loginfo(">>> [AUTO_PICK] Triggering pick operation with consistent pose")
                    self.trigger_pick()
                else:
                    rospy.loginfo(f">>> [AUTO_PICK] Auto-pick disabled or not idle (state: {self.current_state})")
            else:
                rospy.loginfo(f">>> [POSE_BUFFER] Poses not consistent yet - waiting for more consistent poses")
    
    def trigger_pick_callback(self, msg):
        """Callback to trigger pick operation."""
        if msg.data:
            self.trigger_pick()
    
    def trigger_pick(self):
        """Trigger pick operation."""
        rospy.loginfo(">>> [TRIGGER] Attempting to trigger pick operation...")
        if self.current_state != self.STATE_IDLE:
            rospy.logwarn(f">>> [TRIGGER] Cannot trigger pick: current state is {self.current_state}")
            return
        if self.target_pose is None:
            rospy.logwarn(">>> [TRIGGER] Cannot trigger pick: no target pose available")
            return
        self.pick_triggered = True
        rospy.loginfo(">>> [TRIGGER] *** PICK OPERATION TRIGGERED ***")
        rospy.loginfo(f">>> [TRIGGER] Target pose: ({self.target_pose.pose.position.x:.3f}, "
                     f"{self.target_pose.pose.position.y:.3f}, {self.target_pose.pose.position.z:.3f})")
    
    def run(self):
        """Main loop - executes state machine."""
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            if self.current_state == self.STATE_IDLE:
                if self.pick_triggered:
                    rospy.loginfo(">>> [STATE_MACHINE] Transitioning: IDLE -> PREPARING")
                    self.current_state = self.STATE_PREPARING
                    self.pick_triggered = False
                    self._publish_status(self.STATE_PREPARING)
            
            elif self.current_state == self.STATE_PREPARING:
                rospy.loginfo(">>> [STATE_MACHINE] In PREPARING state - executing preparation...")
                if self._prepare_for_pick():
                    rospy.loginfo(">>> [STATE_MACHINE] Transitioning: PREPARING -> APPROACHING")
                    self.current_state = self.STATE_APPROACHING
                    self._publish_status(self.STATE_APPROACHING)
                else:
                    rospy.logerr(">>> [STATE_MACHINE] Transitioning: PREPARING -> FAILED")
                    self.current_state = self.STATE_FAILED
                    self._publish_status(self.STATE_FAILED)
            
            elif self.current_state == self.STATE_APPROACHING:
                rospy.loginfo(">>> [STATE_MACHINE] In APPROACHING state - executing approach...")
                if self._approach_object():
                    rospy.loginfo(">>> [STATE_MACHINE] Transitioning: APPROACHING -> GRASPING")
                    self.current_state = self.STATE_GRASPING
                    self._publish_status(self.STATE_GRASPING)
                else:
                    rospy.logerr(">>> [STATE_MACHINE] Transitioning: APPROACHING -> FAILED")
                    self.current_state = self.STATE_FAILED
                    self._publish_status(self.STATE_FAILED)
            
            elif self.current_state == self.STATE_GRASPING:
                rospy.loginfo(">>> [STATE_MACHINE] In GRASPING state - executing grasp...")
                if self._grasp_object():
                    rospy.loginfo(">>> [STATE_MACHINE] Transitioning: GRASPING -> LIFTING")
                    self.current_state = self.STATE_LIFTING
                    self._publish_status(self.STATE_LIFTING)
                else:
                    rospy.logerr(">>> [STATE_MACHINE] Transitioning: GRASPING -> FAILED")
                    self.current_state = self.STATE_FAILED
                    self._publish_status(self.STATE_FAILED)
            
            elif self.current_state == self.STATE_LIFTING:
                rospy.loginfo(">>> [STATE_MACHINE] In LIFTING state - executing lift...")
                if self._lift_object():
                    rospy.loginfo(">>> [STATE_MACHINE] Transitioning: LIFTING -> COMPLETE")
                    self.current_state = self.STATE_COMPLETE
                    self._publish_status(self.STATE_COMPLETE)
                    self.pick_complete_pub.publish(Bool(True))
                    rospy.loginfo(">>> [STATE_MACHINE] *** PICK OPERATION COMPLETE ***")
                else:
                    rospy.logerr(">>> [STATE_MACHINE] Transitioning: LIFTING -> FAILED")
                    self.current_state = self.STATE_FAILED
                    self._publish_status(self.STATE_FAILED)
            
            elif self.current_state == self.STATE_COMPLETE:
                # Stay in complete state
                pass
            
            elif self.current_state == self.STATE_FAILED:
                # Stay in failed state
                pass
            
            rate.sleep()
    
    def _prepare_for_pick(self):
        """Prepare robot for picking (open gripper, move to safe position)."""
        rospy.loginfo(">>> [PREPARE] Starting preparation...")
        try:
            # Verify current robot state before starting
            rospy.loginfo(">>> [PREPARE] Step 0: Verifying current robot state...")
            try:
                current_joint_values = self.arm.get_current_joint_values()
                current_pose = self.arm.get_current_pose().pose
                rospy.loginfo(f">>> [PREPARE]   Current state - Joint values: {[f'{v:.3f}' for v in current_joint_values]}")
                rospy.loginfo(f">>> [PREPARE]   Current pose: ({current_pose.position.x:.3f}, {current_pose.position.y:.3f}, {current_pose.position.z:.3f})")
            except Exception as e:
                rospy.logwarn(f">>> [PREPARE]   Warning: Could not get current state: {e}")
            
            rospy.loginfo(">>> [PREPARE] Step 1: Opening gripper...")
            self._open_gripper()
            rospy.loginfo(">>> [PREPARE] Step 1 complete: Gripper opened")
            
            try:
                rospy.loginfo(">>> [PREPARE] Step 2: Moving arm to 'go' position...")
                # Get state before movement
                before_joint_values = self.arm.get_current_joint_values()
                before_pose = self.arm.get_current_pose().pose
                joint_names = self.arm.get_active_joints()
                rospy.loginfo(f">>> [PREPARE]   Before movement - Joint values: {dict(zip(joint_names, [f'{v:.3f}' for v in before_joint_values]))}")
                rospy.loginfo(f">>> [PREPARE]   Before movement - Pose: ({before_pose.position.x:.3f}, {before_pose.position.y:.3f}, {before_pose.position.z:.3f})")
                
                # Small delay to ensure state is synchronized after gripper action
                rospy.sleep(0.2)
                self.arm.set_named_target('go')
                self.arm.set_planning_time(10.0)  # Give it time to plan
                rospy.loginfo(">>> [PREPARE]   Executing movement to 'go' position...")
                success = self.arm.go(wait=True)
                
                if success:
                    rospy.loginfo(">>> [PREPARE] Step 2 complete: Arm moved to 'go' position")
                    # Verify final state and check if it actually moved
                    final_joint_values = self.arm.get_current_joint_values()
                    final_pose = self.arm.get_current_pose().pose
                    rospy.loginfo(f">>> [PREPARE]   After movement - Joint values: {dict(zip(joint_names, [f'{v:.3f}' for v in final_joint_values]))}")
                    rospy.loginfo(f">>> [PREPARE]   After movement - Pose: ({final_pose.position.x:.3f}, {final_pose.position.y:.3f}, {final_pose.position.z:.3f})")
                    
                    # Check if joints actually moved
                    joint_moved = False
                    for i, joint_name in enumerate(joint_names):
                        if abs(final_joint_values[i] - before_joint_values[i]) > 0.01:
                            joint_moved = True
                            rospy.loginfo(f">>> [PREPARE]   ✓ {joint_name} moved: {before_joint_values[i]:.3f} -> {final_joint_values[i]:.3f}")
                    
                    if not joint_moved:
                        rospy.logwarn(">>> [PREPARE]   WARNING: No joints moved! This suggests the arm may not be responding to commands.")
                    
                    # Check if pose changed
                    pose_moved = (abs(final_pose.position.x - before_pose.position.x) > 0.01 or
                                 abs(final_pose.position.y - before_pose.position.y) > 0.01 or
                                 abs(final_pose.position.z - before_pose.position.z) > 0.01)
                    if pose_moved:
                        rospy.loginfo(f">>> [PREPARE]   ✓ End-effector moved: ({before_pose.position.x:.3f}, {before_pose.position.y:.3f}, {before_pose.position.z:.3f}) -> ({final_pose.position.x:.3f}, {final_pose.position.y:.3f}, {final_pose.position.z:.3f})")
                    else:
                        rospy.logwarn(">>> [PREPARE]   WARNING: End-effector did not move!")
                else:
                    rospy.logwarn(">>> [PREPARE] Step 2 warning: Failed to move to 'go' position (go() returned False), continuing anyway")
            except Exception as e:
                rospy.logwarn(f">>> [PREPARE] Step 2 warning: Could not move to 'go' position ({e}), continuing anyway")
                import traceback
                rospy.logwarn(traceback.format_exc())
            
            rospy.loginfo(">>> [PREPARE] *** PREPARATION COMPLETE ***")
            return True
        except Exception as e:
            rospy.logerr(f">>> [PREPARE] *** PREPARATION FAILED ***: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return False
    
    def _approach_object(self):
        """Move arm to approach position above object."""
        rospy.loginfo("Approaching object...")
        try:
            # IMPORTANT: Ensure MoveIt has the latest robot state before planning
            # MoveIt automatically gets state from /joint_states topic, but we should verify
            rospy.loginfo(">>> [APPROACH] Step 0: Verifying current robot state...")
            try:
                # Get current state to ensure MoveIt is synchronized
                current_joint_values = self.arm.get_current_joint_values()
                current_pose = self.arm.get_current_pose().pose
                rospy.loginfo(f">>> [APPROACH]   Verified current state - Joint values: {[f'{v:.3f}' for v in current_joint_values]}")
                rospy.loginfo(f">>> [APPROACH]   Verified current pose: ({current_pose.position.x:.3f}, {current_pose.position.y:.3f}, {current_pose.position.z:.3f})")
            except Exception as e:
                rospy.logwarn(f">>> [APPROACH]   Warning: Could not verify current state: {e}")
                rospy.logwarn(">>> [APPROACH]   Continuing anyway - MoveIt should still work")
            
            # Small delay to ensure state is fresh
            rospy.sleep(0.1)
            
            # Transform pose to base_link
            target_pose_base = self._transform_pose_to_base_link(self.target_pose)
            if target_pose_base is None:
                return False
            
            # Check if object is within reach
            target_pos = np.array([
                target_pose_base.pose.position.x,
                target_pose_base.pose.position.y,
                target_pose_base.pose.position.z
            ])
            distance = np.linalg.norm(target_pos)
            horizontal_distance = np.sqrt(target_pos[0]**2 + target_pos[1]**2)
            
            rospy.loginfo(f">>> [APPROACH] Object position in base_link: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
            rospy.loginfo(f">>> [APPROACH]   Distance from base_link: {distance:.3f}m")
            rospy.loginfo(f">>> [APPROACH]   Horizontal distance: {horizontal_distance:.3f}m")
            
            max_reach = 0.65  # HSR arm reach limit
            if distance > max_reach:
                rospy.logwarn(f">>> [APPROACH] WARNING: Object distance ({distance:.3f}m) exceeds max reach ({max_reach}m)")
                # If slightly over, scale it down to bring within reach
                if distance <= max_reach * 1.1:  # Within 10% of limit
                    rospy.loginfo(f">>> [APPROACH]   Scaling target position to bring within reach (scale: {max_reach/distance:.3f})")
                    scale_factor = max_reach / distance
                    target_pos = target_pos * scale_factor
                    rospy.loginfo(f">>> [APPROACH]   Scaled target: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
                    # Update the pose with scaled position
                    target_pose_base.pose.position.x = target_pos[0]
                    target_pose_base.pose.position.y = target_pos[1]
                    target_pose_base.pose.position.z = target_pos[2]
                else:
                    rospy.logwarn(f">>> [APPROACH]   Object too far - may still be reachable depending on arm configuration")
            
            # Calculate approach pose (above object)
            rospy.loginfo(">>> [APPROACH] Step 2: Calculating approach pose...")
            approach_pose = self._calculate_approach_pose(target_pose_base)
            rospy.loginfo(f">>> [APPROACH] Step 2 complete: Approach pose calculated")
            rospy.loginfo(f">>> [APPROACH]   Approach position: ({approach_pose.position.x:.3f}, {approach_pose.position.y:.3f}, {approach_pose.position.z:.3f})")
            
            # Store approach Z for later use in pre-planning
            approach_z = approach_pose.position.z
            
            # Transform to odom for MoveIt
            rospy.loginfo(">>> [APPROACH] Step 3: Transforming approach pose to odom frame...")
            approach_pose_odom = self._transform_pose_to_odom(approach_pose, 'base_link')
            if approach_pose_odom is None:
                rospy.logerr(">>> [APPROACH] Failed to transform approach pose to odom")
                return False
            rospy.loginfo(f">>> [APPROACH] Step 3 complete: Approach pose in odom: ({approach_pose_odom.pose.position.x:.3f}, {approach_pose_odom.pose.position.y:.3f}, {approach_pose_odom.pose.position.z:.3f})")
            
            # Plan and execute
            rospy.loginfo(">>> [APPROACH] Step 4: Setting pose target and planning...")
            rospy.loginfo("="*70)
            rospy.loginfo(">>> [PLAN_DETAILS] *** APPROACH POSE TARGET ***")
            rospy.loginfo(f">>> [PLAN_DETAILS]   Using OBJECT CENTER POSE (not bounding box)")
            rospy.loginfo(f">>> [PLAN_DETAILS]   Target Position (odom): ({approach_pose_odom.pose.position.x:.4f}, {approach_pose_odom.pose.position.y:.4f}, {approach_pose_odom.pose.position.z:.4f})")
            rospy.loginfo(f">>> [PLAN_DETAILS]   Target Orientation (quaternion): ({approach_pose_odom.pose.orientation.x:.4f}, {approach_pose_odom.pose.orientation.y:.4f}, {approach_pose_odom.pose.orientation.z:.4f}, {approach_pose_odom.pose.orientation.w:.4f})")
            
            # Convert quaternion to Euler for readability
            euler = euler_from_quaternion([
                approach_pose_odom.pose.orientation.x,
                approach_pose_odom.pose.orientation.y,
                approach_pose_odom.pose.orientation.z,
                approach_pose_odom.pose.orientation.w
            ])
            rospy.loginfo(f">>> [PLAN_DETAILS]   Target Orientation (Euler RPY, rad): ({euler[0]:.4f}, {euler[1]:.4f}, {euler[2]:.4f})")
            rospy.loginfo(f">>> [PLAN_DETAILS]   Target Orientation (Euler RPY, deg): ({euler[0]*180/math.pi:.2f}°, {euler[1]*180/math.pi:.2f}°, {euler[2]*180/math.pi:.2f}°)")
            rospy.loginfo(f">>> [PLAN_DETAILS]   Approach height offset: {self.approach_height:.3f}m above object center")
            rospy.loginfo("="*70)
            
            # Set pose target
            self.arm.set_pose_target(approach_pose_odom.pose)
            
            # Relax orientation constraints for approach - allow more flexibility
            # Increase orientation tolerance significantly to help with planning
            self.arm.set_goal_orientation_tolerance(0.5)  # Allow ~29 degrees tolerance (increased from 0.2)
            self.arm.set_goal_position_tolerance(0.05)  # Allow 5cm position tolerance (increased from 0.02)
            
            # CRITICAL: Ensure planning timeout is actually set and respected
            # MoveIt sometimes has issues with timeout, so we'll set it multiple times
            self.arm.set_planning_time(self.planning_timeout)
            actual_timeout = self.arm.get_planning_time()
            rospy.loginfo(f">>> [APPROACH]   Planning timeout: {self.planning_timeout}s (actual: {actual_timeout}s)")
            rospy.loginfo(f">>> [APPROACH]   Orientation tolerance: 0.5 rad (~29 deg), Position tolerance: 0.05m")
            
            # Also set workspace bounds if needed - sometimes MoveIt needs explicit bounds
            try:
                # Get current workspace bounds
                workspace = self.arm.get_workspace()
                rospy.loginfo(f">>> [APPROACH]   Workspace bounds: {workspace}")
            except:
                pass  # Workspace bounds might not be available
            
            # Get current arm position before planning
            current_pose_before = self.arm.get_current_pose().pose
            rospy.loginfo(f">>> [APPROACH]   Current arm position BEFORE: ({current_pose_before.position.x:.3f}, {current_pose_before.position.y:.3f}, {current_pose_before.position.z:.3f})")
            
            # Get current joint values with names
            current_joint_values = self.arm.get_current_joint_values()
            joint_names = self.arm.get_active_joints()
            joint_dict = dict(zip(joint_names, [f'{v:.3f}' for v in current_joint_values]))
            rospy.loginfo(f">>> [APPROACH]   Current joint values: {joint_dict}")
            
            # Specifically log arm_lift_joint if available
            if 'arm_lift_joint' in joint_names:
                lift_idx = joint_names.index('arm_lift_joint')
                rospy.loginfo(f">>> [APPROACH]   arm_lift_joint (vertical lift): {current_joint_values[lift_idx]:.3f} m")
            
            # Publish current end effector marker
            current_pose_before_stamped = PoseStamped()
            current_pose_before_stamped.header.frame_id = 'odom'
            current_pose_before_stamped.header.stamp = rospy.Time.now()
            current_pose_before_stamped.pose = current_pose_before
            self._publish_end_effector_marker(current_pose_before_stamped, color=(1.0, 1.0, 0.0, 1.0), scale=0.06)
            
            # Try multiple planning strategies for better success rate
            plan_success = False
            trajectory = None
            planning_time = None
            error_code = None
            
            # Ensure planning timeout is set before Strategy 1
            # CRITICAL: MoveIt sometimes ignores set_planning_time, so we set it multiple times
            self.arm.set_planning_time(self.planning_timeout)
            rospy.sleep(0.1)  # Brief pause to ensure timeout is set
            self.arm.set_planning_time(self.planning_timeout)  # Set again to be sure
            actual_timeout = self.arm.get_planning_time()
            if abs(actual_timeout - self.planning_timeout) > 0.1:
                rospy.logwarn(f">>> [APPROACH]   WARNING: Planning timeout mismatch! Requested: {self.planning_timeout}s, Actual: {actual_timeout}s")
                # Force it again
                self.arm.set_planning_time(self.planning_timeout)
            
            # Pre-planning: Try to set arm_lift_joint to a reasonable value for top-down picking
            # Use a simpler, more direct approach - just set the joint value directly
            joint_names = self.arm.get_active_joints()
            if 'arm_lift_joint' in joint_names:
                current_joint_values = self.arm.get_current_joint_values()
                lift_idx = joint_names.index('arm_lift_joint')
                current_lift = current_joint_values[lift_idx]
                
                # For top-down picking, we want arm_lift_joint to be at least 0.3m (mid-range)
                # This gives good vertical clearance while maintaining reach
                target_lift = max(0.3, min(0.5, approach_z - 0.2))  # Adjust based on target Z
                target_lift = min(0.69, target_lift)  # Respect upper limit
                
                if abs(current_lift - target_lift) > 0.05:  # If more than 5cm difference
                    rospy.loginfo(f">>> [APPROACH]   Pre-setting arm_lift_joint: {current_lift:.3f}m -> {target_lift:.3f}m")
                    try:
                        # Use a simpler approach: set only arm_lift_joint and use go() with shorter timeout
                        self.arm.set_joint_value_target({'arm_lift_joint': target_lift})
                        # Use a shorter timeout for this simple joint movement
                        original_timeout = self.arm.get_planning_time()
                        self.arm.set_planning_time(5.0)  # 5 seconds should be enough for single joint
                        rospy.loginfo(">>> [APPROACH]   Adjusting arm_lift_joint before approach planning...")
                        success = self.arm.go(wait=True)
                        self.arm.set_planning_time(original_timeout)  # Restore original timeout
                        if success:
                            rospy.sleep(0.5)  # Brief pause for state update
                            rospy.loginfo(">>> [APPROACH]   arm_lift_joint adjusted successfully")
                        else:
                            rospy.logwarn(">>> [APPROACH]   Could not adjust arm_lift_joint, continuing with current value")
                    except Exception as e:
                        rospy.logwarn(f">>> [APPROACH]   Failed to pre-set arm_lift_joint: {e}")
                        # Continue anyway - MoveIt should still work
            
            # Strategy 1: Try direct pose planning with current planner
            rospy.loginfo(">>> [APPROACH]   Strategy 1: Direct pose planning with RRTConnect...")
            # Ensure timeout is set right before planning
            self.arm.set_planning_time(self.planning_timeout)
            rospy.loginfo(f">>> [APPROACH]   Planning timeout before Strategy 1: {self.arm.get_planning_time()}s")
            plan_result = self.arm.plan()
            if len(plan_result) >= 2:
                plan_success, trajectory, planning_time, error_code = plan_result[0], plan_result[1], plan_result[2] if len(plan_result) > 2 else None, plan_result[3] if len(plan_result) > 3 else None
            else:
                plan_success = plan_result[0] if plan_result else False
                trajectory = plan_result[1] if len(plan_result) > 1 else None
                planning_time = plan_result[2] if len(plan_result) > 2 else None
                error_code = plan_result[3] if len(plan_result) > 3 else None
            
            # Strategy 2: If direct planning fails, try Cartesian path planning (simpler, more reliable)
            if not (plan_success and trajectory is not None):
                rospy.logwarn(">>> [APPROACH]   Strategy 1 failed, trying Strategy 2: Cartesian path planning...")
                try:
                    # Compute Cartesian path (straight line motion)
                    # Correct signature: compute_cartesian_path(waypoints, eef_step, avoid_collisions)
                    waypoints = [approach_pose_odom.pose]
                    (plan_cartesian, fraction) = self.arm.compute_cartesian_path(waypoints, 0.01, False)
                    if fraction >= 0.9:  # At least 90% of path is valid
                        rospy.loginfo(f">>> [APPROACH]   Cartesian path computed: {fraction*100:.1f}% valid")
                        plan_success = True
                        trajectory = plan_cartesian
                    else:
                        rospy.logwarn(f">>> [APPROACH]   Cartesian path only {fraction*100:.1f}% valid, trying alternative planners...")
                except Exception as e:
                    rospy.logwarn(f">>> [APPROACH]   Cartesian path planning failed: {e}")
            
            # Strategy 3: Try alternative planners if previous strategies failed
            if not (plan_success and trajectory is not None):
                # Use longer timeout for alternative planners - they might need more time
                original_timeout = self.planning_timeout
                self.arm.set_planning_time(30.0)  # Give alternative planners more time
                
                alternative_planners = ['RRT', 'PRM', 'RRTstar', 'EST', 'SBL']
                for planner_name in alternative_planners:
                    rospy.logwarn(f">>> [APPROACH]   Strategy 3: Trying alternative planner: {planner_name} (timeout: 30s)...")
                    try:
                        self.arm.set_planner_id(planner_name)
                        # Ensure timeout is set for this planner
                        self.arm.set_planning_time(30.0)
                        plan_result = self.arm.plan()
                        if len(plan_result) >= 2:
                            plan_success, trajectory = plan_result[0], plan_result[1]
                        else:
                            plan_success = plan_result[0] if plan_result else False
                            trajectory = plan_result[1] if len(plan_result) > 1 else None
                        
                        if plan_success and trajectory is not None:
                            rospy.loginfo(f">>> [APPROACH]   Success with planner: {planner_name}")
                            break
                    except Exception as e:
                        rospy.logwarn(f">>> [APPROACH]   Planner {planner_name} failed: {e}")
                        continue
                
                # Restore original timeout and planner
                self.arm.set_planning_time(original_timeout)
                self.arm.set_planner_id('RRTConnect')
            
            # Strategy 4: If all planning fails, try with maximum relaxed constraints
            if not (plan_success and trajectory is not None):
                rospy.logwarn(">>> [APPROACH]   Strategy 4: Trying with maximum relaxed constraints...")
                # Maximum relaxation - allow any orientation, larger position tolerance
                self.arm.set_goal_orientation_tolerance(1.57)  # ~90 degrees (almost any orientation)
                self.arm.set_goal_position_tolerance(0.10)  # 10cm position tolerance
                self.arm.set_planning_time(30.0)  # Give it more time with relaxed constraints
                try:
                    plan_result = self.arm.plan()
                    if len(plan_result) >= 2:
                        plan_success, trajectory = plan_result[0], plan_result[1]
                    else:
                        plan_success = plan_result[0] if plan_result else False
                        trajectory = plan_result[1] if len(plan_result) > 1 else None
                    
                    if plan_success and trajectory is not None:
                        rospy.logwarn(">>> [APPROACH]   Success with maximum relaxed constraints")
                except Exception as e:
                    rospy.logwarn(f">>> [APPROACH]   Maximum relaxation also failed: {e}")
                
                # Restore reasonable constraints
                self.arm.set_goal_orientation_tolerance(0.5)
                self.arm.set_goal_position_tolerance(0.05)
                self.arm.set_planning_time(self.planning_timeout)
            
            # Strategy 5: Try planning to an intermediate waypoint (halfway between current and target)
            if not (plan_success and trajectory is not None):
                rospy.logwarn(">>> [APPROACH]   Strategy 5: Trying intermediate waypoint approach...")
                try:
                    # Calculate intermediate pose (halfway between current and target)
                    current_pos = np.array([
                        current_pose_before.position.x,
                        current_pose_before.position.y,
                        current_pose_before.position.z
                    ])
                    target_pos = np.array([
                        approach_pose_odom.pose.position.x,
                        approach_pose_odom.pose.position.y,
                        approach_pose_odom.pose.position.z
                    ])
                    intermediate_pos = (current_pos + target_pos) / 2.0
                    
                    # Create intermediate pose with same orientation as target
                    intermediate_pose = approach_pose_odom.pose
                    intermediate_pose.position.x = intermediate_pos[0]
                    intermediate_pose.position.y = intermediate_pos[1]
                    intermediate_pose.position.z = intermediate_pos[2]
                    
                    rospy.loginfo(f">>> [APPROACH]   Intermediate waypoint: ({intermediate_pos[0]:.3f}, {intermediate_pos[1]:.3f}, {intermediate_pos[2]:.3f})")
                    
                    # Try planning to intermediate first
                    self.arm.set_pose_target(intermediate_pose)
                    self.arm.set_goal_orientation_tolerance(0.5)
                    self.arm.set_goal_position_tolerance(0.10)
                    self.arm.set_planning_time(10.0)
                    
                    plan_result = self.arm.plan()
                    if len(plan_result) >= 2:
                        intermediate_success, intermediate_trajectory = plan_result[0], plan_result[1]
                    else:
                        intermediate_success = plan_result[0] if plan_result else False
                        intermediate_trajectory = plan_result[1] if len(plan_result) > 1 else None
                    
                    if intermediate_success and intermediate_trajectory is not None:
                        rospy.loginfo(">>> [APPROACH]   Successfully planned to intermediate waypoint, executing...")
                        execution_result = self.arm.execute(intermediate_trajectory, wait=True)
                        if execution_result:
                            rospy.loginfo(">>> [APPROACH]   Reached intermediate waypoint, now planning to final target...")
                            rospy.sleep(0.5)  # Brief pause for state update
                            
                            # Now try planning to final target from intermediate position
                            self.arm.set_pose_target(approach_pose_odom.pose)
                            self.arm.set_goal_orientation_tolerance(0.5)
                            self.arm.set_goal_position_tolerance(0.05)
                            self.arm.set_planning_time(self.planning_timeout)
                            
                            plan_result = self.arm.plan()
                            if len(plan_result) >= 2:
                                plan_success, trajectory = plan_result[0], plan_result[1]
                            else:
                                plan_success = plan_result[0] if plan_result else False
                                trajectory = plan_result[1] if len(plan_result) > 1 else None
                            
                            if plan_success and trajectory is not None:
                                rospy.loginfo(">>> [APPROACH]   Successfully planned from intermediate to final target")
                            else:
                                rospy.logwarn(">>> [APPROACH]   Could not plan from intermediate to final target")
                        else:
                            rospy.logwarn(">>> [APPROACH]   Failed to execute intermediate trajectory")
                    else:
                        rospy.logwarn(">>> [APPROACH]   Could not plan to intermediate waypoint")
                except Exception as e:
                    rospy.logwarn(f">>> [APPROACH]   Intermediate waypoint strategy failed: {e}")
                    import traceback
                    rospy.logwarn(traceback.format_exc())
            
            if plan_success and trajectory is not None:
                rospy.loginfo("="*70)
                rospy.loginfo(f">>> [PLAN_DETAILS] *** PLANNING SUCCEEDED ***")
                rospy.loginfo(f">>> [PLAN_DETAILS]   Trajectory has {len(trajectory.joint_trajectory.points)} waypoints")
                if planning_time is not None:
                    rospy.loginfo(f">>> [PLAN_DETAILS]   Planning time: {planning_time:.3f}s")
                if error_code is not None:
                    rospy.loginfo(f">>> [PLAN_DETAILS]   Error code: {error_code}")
                
                # Print trajectory details
                if len(trajectory.joint_trajectory.points) > 0:
                    first_point = trajectory.joint_trajectory.points[0]
                    last_point = trajectory.joint_trajectory.points[-1]
                    rospy.loginfo(f">>> [PLAN_DETAILS]   First waypoint joint values: {[f'{v:.3f}' for v in first_point.positions]}")
                    rospy.loginfo(f">>> [PLAN_DETAILS]   Last waypoint joint values: {[f'{v:.3f}' for v in last_point.positions]}")
                    if hasattr(first_point, 'time_from_start') and hasattr(last_point, 'time_from_start'):
                        total_time = (last_point.time_from_start - first_point.time_from_start).to_sec()
                        rospy.loginfo(f">>> [PLAN_DETAILS]   Estimated execution time: {total_time:.2f}s")
                    
                    # Calculate end-effector pose from last waypoint
                    rospy.loginfo(">>> [PLAN_DETAILS]   Calculating expected end-effector pose from trajectory...")
                    # Set joint values to last waypoint to get expected pose
                    self.arm.set_joint_value_target(last_point.positions)
                    expected_pose = self.arm.get_current_pose().pose  # This gets the pose for the target, not current
                    # Actually, we need to compute forward kinematics - but MoveIt doesn't expose this easily
                    # Instead, just log that we're planning to reach the target pose
                    rospy.loginfo(f">>> [PLAN_DETAILS]   Expected final position: ({approach_pose_odom.pose.position.x:.4f}, {approach_pose_odom.pose.position.y:.4f}, {approach_pose_odom.pose.position.z:.4f})")
                
                rospy.loginfo("="*70)
                
                # Publish visualization markers
                # Publish path marker (line from current to target)
                current_pose_before_stamped = PoseStamped()
                current_pose_before_stamped.header.frame_id = 'odom'
                current_pose_before_stamped.header.stamp = rospy.Time.now()
                current_pose_before_stamped.pose = current_pose_before
                self._publish_path_marker(current_pose_before, approach_pose_odom, frame_id='odom', color=(0.0, 0.0, 1.0, 1.0), scale=0.03)
                
                # Publish end effector marker at target
                self._publish_end_effector_marker(approach_pose_odom, color=(0.0, 1.0, 0.0, 1.0), scale=0.08)
                
                # Execute the planned trajectory
                rospy.loginfo(">>> [APPROACH]   Executing planned trajectory...")
                execution_result = self.arm.execute(trajectory, wait=True)
                
                # Check arm position after execution
                current_pose_after = self.arm.get_current_pose().pose
                rospy.loginfo(f">>> [APPROACH]   Current arm position AFTER: ({current_pose_after.position.x:.3f}, {current_pose_after.position.y:.3f}, {current_pose_after.position.z:.3f})")
                
                # Publish end effector marker at current position
                current_pose_after_stamped = PoseStamped()
                current_pose_after_stamped.header.frame_id = 'odom'
                current_pose_after_stamped.header.stamp = rospy.Time.now()
                current_pose_after_stamped.pose = current_pose_after
                self._publish_end_effector_marker(current_pose_after_stamped, color=(0.0, 0.5, 0.5, 1.0), scale=0.06)
                
                if execution_result:
                    rospy.loginfo(">>> [APPROACH]   *** EXECUTION SUCCEEDED ***")
                    rospy.loginfo(">>> [APPROACH] *** APPROACH COMPLETE ***")
                    return True
                else:
                    rospy.logerr(">>> [APPROACH]   *** EXECUTION FAILED ***")
                    rospy.logerr(">>> [APPROACH]   Execution returned False - trajectory may have been aborted")
                    return False
            else:
                rospy.logerr(">>> [APPROACH]   *** PLANNING FAILED ***")
                if error_code is not None:
                    rospy.logerr(f">>> [APPROACH]   Error code: {error_code}")
                
                # Strategy 6: Try joint space planning - sometimes pose planning fails but joint space works
                rospy.logwarn(">>> [APPROACH]   Strategy 6: Trying joint space planning as fallback...")
                try:
                    # Try to get IK solution for target pose
                    self.arm.set_pose_target(approach_pose_odom.pose)
                    # Use async planning with longer timeout
                    self.arm.set_planning_time(60.0)  # Give it a full minute
                    self.arm.set_goal_orientation_tolerance(1.57)  # Maximum orientation tolerance
                    self.arm.set_goal_position_tolerance(0.10)  # Larger position tolerance
                    
                    # Try planning again with very relaxed constraints
                    plan_result = self.arm.plan()
                    if len(plan_result) >= 2:
                        plan_success, trajectory = plan_result[0], plan_result[1]
                    else:
                        plan_success = plan_result[0] if plan_result else False
                        trajectory = plan_result[1] if len(plan_result) > 1 else None
                    
                    if plan_success and trajectory is not None:
                        rospy.logwarn(">>> [APPROACH]   Success with joint space fallback!")
                    else:
                        rospy.logwarn(">>> [APPROACH]   Joint space planning also failed")
                except Exception as e:
                    rospy.logwarn(f">>> [APPROACH]   Joint space planning failed: {e}")
                
                # Final fallback: try go() method with very relaxed constraints
                if not (plan_success and trajectory is not None):
                    rospy.logwarn(">>> [APPROACH]   Trying final fallback: arm.go() method with relaxed constraints...")
                    self.arm.set_goal_orientation_tolerance(1.57)
                    self.arm.set_goal_position_tolerance(0.10)
                    self.arm.set_planning_time(60.0)
                    rospy.loginfo(">>> [APPROACH]   Executing arm.go() - this may take a while...")
                    success = self.arm.go(wait=True)
                
                    # Check arm position after execution
                    current_pose_after = self.arm.get_current_pose().pose
                    rospy.loginfo(f">>> [APPROACH]   Current arm position AFTER: ({current_pose_after.position.x:.3f}, {current_pose_after.position.y:.3f}, {current_pose_after.position.z:.3f})")
                    
                    if success:
                        rospy.loginfo(">>> [APPROACH] *** APPROACH COMPLETE (via go() fallback) ***")
                        return True
                    else:
                        rospy.logerr(">>> [APPROACH] Failed to approach object - all planning strategies failed")
                        return False
                else:
                    # We already have a successful plan from Strategy 6
                    rospy.loginfo(">>> [APPROACH]   Executing trajectory from Strategy 6...")
                    execution_result = self.arm.execute(trajectory, wait=True)
                    current_pose_after = self.arm.get_current_pose().pose
                    rospy.loginfo(f">>> [APPROACH]   Current arm position AFTER: ({current_pose_after.position.x:.3f}, {current_pose_after.position.y:.3f}, {current_pose_after.position.z:.3f})")
                    
                    if execution_result:
                        rospy.loginfo(">>> [APPROACH] *** APPROACH COMPLETE (via Strategy 6) ***")
                        return True
                    else:
                        rospy.logerr(">>> [APPROACH] Failed to execute trajectory from Strategy 6")
                        return False
        except Exception as e:
            rospy.logerr(f"Failed to approach: {e}")
            return False
    
    def _grasp_object(self):
        """Move arm down to grasp position and close gripper."""
        rospy.loginfo(">>> [GRASP] Starting grasp operation...")
        try:
            rospy.loginfo(">>> [GRASP] Step 1: Transforming pose to base_link...")
            target_pose_base = self._transform_pose_to_base_link(self.target_pose)
            if target_pose_base is None:
                rospy.logerr(">>> [GRASP] Step 1 FAILED: Could not transform pose to base_link")
                return False
            rospy.loginfo(">>> [GRASP] Step 1 complete: Pose transformed to base_link")
            
            rospy.loginfo(">>> [GRASP] Step 2: Calculating grasp pose...")
            grasp_pose = self._calculate_grasp_pose(target_pose_base)
            rospy.loginfo(f">>> [GRASP] Step 2 complete: Grasp pose calculated")
            rospy.loginfo(f">>> [GRASP]   Grasp position: ({grasp_pose.position.x:.3f}, {grasp_pose.position.y:.3f}, {grasp_pose.position.z:.3f})")
            
            rospy.loginfo(">>> [GRASP] Step 3: Transforming grasp pose to odom...")
            grasp_pose_odom = self._transform_pose_to_odom(grasp_pose, 'base_link')
            if grasp_pose_odom is None:
                rospy.logerr(">>> [GRASP] Step 3 FAILED: Could not transform to odom")
                return False
            rospy.loginfo(f">>> [GRASP] Step 3 complete: Grasp pose in odom frame")
            
            rospy.loginfo(">>> [GRASP] Step 4: Moving arm down to grasp position...")
            
            rospy.loginfo("="*70)
            rospy.loginfo(">>> [PLAN_DETAILS] *** GRASP POSE TARGET ***")
            rospy.loginfo(f">>> [PLAN_DETAILS]   Using OBJECT TOP SURFACE (calculated from center + height/2)")
            rospy.loginfo(f">>> [PLAN_DETAILS]   Target Position (odom): ({grasp_pose_odom.pose.position.x:.4f}, {grasp_pose_odom.pose.position.y:.4f}, {grasp_pose_odom.pose.position.z:.4f})")
            rospy.loginfo(f">>> [PLAN_DETAILS]   Target Orientation (quaternion): ({grasp_pose_odom.pose.orientation.x:.4f}, {grasp_pose_odom.pose.orientation.y:.4f}, {grasp_pose_odom.pose.orientation.z:.4f}, {grasp_pose_odom.pose.orientation.w:.4f})")
            
            # Convert quaternion to Euler for readability
            euler = euler_from_quaternion([
                grasp_pose_odom.pose.orientation.x,
                grasp_pose_odom.pose.orientation.y,
                grasp_pose_odom.pose.orientation.z,
                grasp_pose_odom.pose.orientation.w
            ])
            rospy.loginfo(f">>> [PLAN_DETAILS]   Target Orientation (Euler RPY, deg): ({euler[0]*180/math.pi:.2f}°, {euler[1]*180/math.pi:.2f}°, {euler[2]*180/math.pi:.2f}°)")
            rospy.loginfo(f">>> [PLAN_DETAILS]   Grasp height offset: {self.grasp_height_offset:.3f}m above object TOP SURFACE")
            rospy.loginfo(f">>> [PLAN_DETAILS]   End-effector orientation (from config): {self.end_effector_orientation}")
            rospy.loginfo("="*70)
            
            # Get current arm position before planning
            current_pose_before = self.arm.get_current_pose().pose
            rospy.loginfo(f">>> [GRASP]   Current arm position BEFORE: ({current_pose_before.position.x:.3f}, {current_pose_before.position.y:.3f}, {current_pose_before.position.z:.3f})")
            
            # Calculate distance to target
            distance_to_target = np.sqrt(
                (current_pose_before.position.x - grasp_pose_odom.pose.position.x)**2 +
                (current_pose_before.position.y - grasp_pose_odom.pose.position.y)**2 +
                (current_pose_before.position.z - grasp_pose_odom.pose.position.z)**2
            )
            rospy.loginfo(f">>> [GRASP]   Distance to target: {distance_to_target:.3f}m")
            
            # Get current joint values
            current_joint_values = self.arm.get_current_joint_values()
            rospy.loginfo(f">>> [GRASP]   Current joint values: {[f'{v:.3f}' for v in current_joint_values]}")
            
            # Try Cartesian path first for downward motion (simpler, more reliable)
            plan_success = False
            trajectory = None
            
            # Strategy 1: Try Cartesian path for straight downward motion
            if distance_to_target < 0.20:  # Only if close (within 20cm)
                rospy.loginfo(">>> [GRASP]   Strategy 1: Trying Cartesian path for downward motion...")
                try:
                    waypoints = [grasp_pose_odom.pose]
                    (plan_cartesian, fraction) = self.arm.compute_cartesian_path(waypoints, 0.01, False)
                    if fraction >= 0.9:
                        rospy.loginfo(f">>> [GRASP]   Cartesian path computed: {fraction*100:.1f}% valid")
                        plan_success = True
                        trajectory = plan_cartesian
                    else:
                        rospy.logwarn(f">>> [GRASP]   Cartesian path only {fraction*100:.1f}% valid, trying pose planning...")
                except Exception as e:
                    rospy.logwarn(f">>> [GRASP]   Cartesian path planning failed: {e}")
            
            # Strategy 2: Pose-based planning with relaxed constraints
            if not (plan_success and trajectory is not None):
                rospy.loginfo(">>> [GRASP]   Strategy 2: Trying pose-based planning...")
                self.arm.set_pose_target(grasp_pose_odom.pose)
                # More relaxed constraints for grasp (we're already close)
                self.arm.set_goal_orientation_tolerance(0.5)  # Increased from 0.2
                self.arm.set_goal_position_tolerance(0.05)  # Increased from 0.02
                
                plan_result = self.arm.plan()
                if len(plan_result) >= 2:
                    plan_success, trajectory = plan_result[0], plan_result[1]
                else:
                    plan_success = plan_result[0] if plan_result else False
                    trajectory = plan_result[1] if len(plan_result) > 1 else None
            
            if plan_success and trajectory is not None:
                rospy.loginfo("="*70)
                rospy.loginfo(f">>> [PLAN_DETAILS] *** PLANNING SUCCEEDED ***")
                rospy.loginfo(f">>> [PLAN_DETAILS]   Trajectory has {len(trajectory.joint_trajectory.points)} waypoints")
                if len(trajectory.joint_trajectory.points) > 0:
                    first_point = trajectory.joint_trajectory.points[0]
                    last_point = trajectory.joint_trajectory.points[-1]
                    rospy.loginfo(f">>> [PLAN_DETAILS]   First waypoint joint values: {[f'{v:.3f}' for v in first_point.positions]}")
                    rospy.loginfo(f">>> [PLAN_DETAILS]   Last waypoint joint values: {[f'{v:.3f}' for v in last_point.positions]}")
                    if hasattr(first_point, 'time_from_start') and hasattr(last_point, 'time_from_start'):
                        total_time = (last_point.time_from_start - first_point.time_from_start).to_sec()
                        rospy.loginfo(f">>> [PLAN_DETAILS]   Estimated execution time: {total_time:.2f}s")
                rospy.loginfo("="*70)
                rospy.loginfo(">>> [GRASP]   Executing planned trajectory...")
                execution_result = self.arm.execute(trajectory, wait=True)
                
                # Check arm position after execution
                current_pose_after = self.arm.get_current_pose().pose
                rospy.loginfo(f">>> [GRASP]   Current arm position AFTER: ({current_pose_after.position.x:.3f}, {current_pose_after.position.y:.3f}, {current_pose_after.position.z:.3f})")
                
                if not execution_result:
                    rospy.logerr(">>> [GRASP]   *** EXECUTION FAILED ***")
                    rospy.logerr(">>> [GRASP] Step 4 FAILED: Execution returned False - trajectory may have been aborted")
                    return False
            else:
                rospy.logwarn(">>> [GRASP]   Planning failed, trying fallback: arm.go() method...")
                success = self.arm.go(wait=True)
                
                # Check arm position after execution
                current_pose_after = self.arm.get_current_pose().pose
                rospy.loginfo(f">>> [GRASP]   Current arm position AFTER: ({current_pose_after.position.x:.3f}, {current_pose_after.position.y:.3f}, {current_pose_after.position.z:.3f})")
                
                if not success:
                    rospy.logerr(">>> [GRASP] Step 4 FAILED: Failed to move to grasp position")
                    return False
            
            rospy.loginfo(">>> [GRASP] Step 4 complete: Arm moved to grasp position")
            
            rospy.loginfo(">>> [GRASP] Step 5: Waiting for arm to settle (0.5s)...")
            rospy.sleep(0.5)
            
            rospy.loginfo(">>> [GRASP] Step 6: Closing gripper...")
            self._close_gripper()
            rospy.loginfo(">>> [GRASP] Step 6 complete: Gripper closed")
            
            rospy.loginfo(">>> [GRASP] Step 7: Waiting for gripper to fully close (1.0s)...")
            rospy.sleep(1.0)
            
            rospy.loginfo(">>> [GRASP] *** GRASP COMPLETE ***")
            return True
        except Exception as e:
            rospy.logerr(f">>> [GRASP] *** GRASP FAILED ***: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return False
    
    def _lift_object(self):
        """Lift object up after grasping."""
        rospy.loginfo(">>> [LIFT] Starting lift operation...")
        try:
            rospy.loginfo(">>> [LIFT] Step 1: Getting current arm pose...")
            current_pose = self.arm.get_current_pose().pose
            rospy.loginfo(f">>> [LIFT]   Current position: ({current_pose.position.x:.3f}, {current_pose.position.y:.3f}, {current_pose.position.z:.3f})")
            
            rospy.loginfo(">>> [LIFT] Step 2: Calculating lift pose...")
            lift_pose = current_pose
            lift_pose.position.z += self.lift_height
            rospy.loginfo(f">>> [LIFT]   Lift height offset: {self.lift_height:.3f}m")
            rospy.loginfo(f">>> [LIFT]   Target lift position: ({lift_pose.position.x:.3f}, {lift_pose.position.y:.3f}, {lift_pose.position.z:.3f})")
            
            rospy.loginfo(">>> [LIFT] Step 3: Moving arm up to lift position...")
            rospy.loginfo(f">>> [LIFT]   Current arm position BEFORE: ({current_pose.position.x:.3f}, {current_pose.position.y:.3f}, {current_pose.position.z:.3f})")
            
            self.arm.set_pose_target(lift_pose)
            self.arm.set_goal_orientation_tolerance(0.2)
            self.arm.set_goal_position_tolerance(0.02)
            
            rospy.loginfo(">>> [LIFT]   Attempting to plan trajectory...")
            plan_result = self.arm.plan()
            if len(plan_result) >= 2:
                plan_success, trajectory = plan_result[0], plan_result[1]
            else:
                plan_success = plan_result[0] if plan_result else False
                trajectory = plan_result[1] if len(plan_result) > 1 else None
            
            if plan_success and trajectory is not None:
                rospy.loginfo(f">>> [LIFT]   *** PLANNING SUCCEEDED *** ({len(trajectory.joint_trajectory.points)} waypoints)")
                rospy.loginfo(">>> [LIFT]   Executing planned trajectory...")
                execution_result = self.arm.execute(trajectory, wait=True)
                
                # Check arm position after execution
                current_pose_after = self.arm.get_current_pose().pose
                rospy.loginfo(f">>> [LIFT]   Current arm position AFTER: ({current_pose_after.position.x:.3f}, {current_pose_after.position.y:.3f}, {current_pose_after.position.z:.3f})")
                
                if execution_result:
                    rospy.loginfo(">>> [LIFT] *** LIFT COMPLETE ***")
                    return True
                else:
                    rospy.logerr(">>> [LIFT]   *** EXECUTION FAILED ***")
                    rospy.logerr(">>> [LIFT] *** LIFT FAILED *** - execution returned False")
                    return False
            else:
                rospy.logwarn(">>> [LIFT]   Planning failed, trying fallback: arm.go() method...")
                success = self.arm.go(wait=True)
                
                # Check arm position after execution
                current_pose_after = self.arm.get_current_pose().pose
                rospy.loginfo(f">>> [LIFT]   Current arm position AFTER: ({current_pose_after.position.x:.3f}, {current_pose_after.position.y:.3f}, {current_pose_after.position.z:.3f})")
                
                if success:
                    rospy.loginfo(">>> [LIFT] *** LIFT COMPLETE ***")
                    return True
                else:
                    rospy.logerr(">>> [LIFT] *** LIFT FAILED *** - planning or execution failed")
                    return False
        except Exception as e:
            rospy.logerr(f">>> [LIFT] *** LIFT FAILED ***: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return False
    
    def _transform_pose_to_base_link(self, pose_stamped):
        """Transform pose to base_link frame."""
        try:
            rospy.loginfo(f">>> [TRANSFORM] Transforming pose from {pose_stamped.header.frame_id} to base_link")
            rospy.loginfo(f">>> [TRANSFORM]   Original pose: ({pose_stamped.pose.position.x:.3f}, {pose_stamped.pose.position.y:.3f}, {pose_stamped.pose.position.z:.3f})")
            
            if pose_stamped.header.frame_id == 'base_link':
                rospy.loginfo(">>> [TRANSFORM] Pose already in base_link frame")
                return pose_stamped
            
            # Use latest transform (rospy.Time(0)) instead of pose timestamp to avoid extrapolation errors
            # The pose position is still valid, we just need the current transform
            rospy.loginfo(">>> [TRANSFORM]   Using latest transform (rospy.Time(0)) to avoid timestamp issues")
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                pose_stamped.header.frame_id,
                rospy.Time(0),  # Use latest available transform
                timeout=rospy.Duration(1.0)
            )
            
            # Create a new pose with current timestamp for transformation
            pose_for_transform = PoseStamped()
            pose_for_transform.header.frame_id = pose_stamped.header.frame_id
            pose_for_transform.header.stamp = rospy.Time.now()  # Use current time
            pose_for_transform.pose = pose_stamped.pose  # Use the pose data
            
            pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_for_transform, transform)
            pose_transformed.header.stamp = rospy.Time.now()  # Update timestamp
            rospy.loginfo(f">>> [TRANSFORM]   Transformed pose: ({pose_transformed.pose.position.x:.3f}, {pose_transformed.pose.position.y:.3f}, {pose_transformed.pose.position.z:.3f})")
            
            return pose_transformed
        except Exception as e:
            rospy.logerr(f">>> [TRANSFORM] Failed to transform to base_link: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return None
    
    def _transform_pose_to_odom(self, pose, source_frame):
        """Transform pose to odom frame for MoveIt."""
        try:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = source_frame
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.pose = pose
            
            transform = self.tf_buffer.lookup_transform('odom', source_frame, rospy.Time.now(), timeout=rospy.Duration(1.0))
            pose_odom = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
            return pose_odom
        except Exception as e:
            rospy.logerr(f"Failed to transform to odom: {e}")
            return None
    
    def _calculate_approach_pose(self, target_pose_base):
        """Calculate approach pose (above object TOP SURFACE, not center).
        
        For top-down picking, we want to ensure the arm_lift_joint is used to raise
        the arm vertically. The approach Z should be high enough to allow the arm
        to reach over the object.
        """
        # The target_pose_base is the object's CENTER position
        center_pos = np.array([
            target_pose_base.pose.position.x,
            target_pose_base.pose.position.y,
            target_pose_base.pose.position.z
        ])
        
        # Calculate top surface position: center_Z + (height / 2)
        # In ROS, Z is up, so we add half the height to get to the top
        top_surface_z = center_pos[2] + (self.object_height / 2.0)
        
        # Approach position: top surface + approach_height
        approach_pos = center_pos.copy()
        approach_pos[2] = top_surface_z + self.approach_height
        
        # For top-down picking, ensure minimum height to allow arm_lift_joint usage
        # arm_lift_joint range: 0.0 to 0.69m, arm base is at ~0.34m from base_link
        # So total reachable height from base_link is ~1.03m (0.34 + 0.69)
        # Ensure approach Z is at least 0.5m from base_link to encourage arm_lift usage
        min_approach_z_from_base = 0.5
        if approach_pos[2] < min_approach_z_from_base:
            rospy.loginfo(f">>> [APPROACH]   Approach Z ({approach_pos[2]:.3f}m) is below minimum ({min_approach_z_from_base:.3f}m)")
            rospy.loginfo(f">>> [APPROACH]   Adjusting to minimum to encourage arm_lift_joint usage")
            approach_pos[2] = min_approach_z_from_base
        
        rospy.loginfo(f">>> [APPROACH]   Object center Z: {center_pos[2]:.3f}m")
        rospy.loginfo(f">>> [APPROACH]   Object height: {self.object_height:.3f}m")
        rospy.loginfo(f">>> [APPROACH]   Top surface Z: {top_surface_z:.3f}m (center + height/2)")
        rospy.loginfo(f">>> [APPROACH]   Approach Z: {approach_pos[2]:.3f}m (top + {self.approach_height:.3f}m, min: {min_approach_z_from_base:.3f}m)")
        
        # Use top-down orientation (same as grasp pose) for consistent planning
        # This ensures the arm can smoothly transition from approach to grasp
        roll, pitch, yaw = self.end_effector_orientation
        quat = quaternion_from_euler(roll, pitch, yaw)
        
        pose = self.arm.get_current_pose().pose  # Get pose structure
        pose.position.x = approach_pos[0]
        pose.position.y = approach_pos[1]
        pose.position.z = approach_pos[2]
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        
        return pose
    
    def _calculate_grasp_pose(self, target_pose_base):
        """Calculate grasp pose (at object TOP SURFACE with offset, not center)."""
        # The target_pose_base is the object's CENTER position
        center_pos = np.array([
            target_pose_base.pose.position.x,
            target_pose_base.pose.position.y,
            target_pose_base.pose.position.z
        ])
        
        # Calculate top surface position: center_Z + (height / 2)
        top_surface_z = center_pos[2] + (self.object_height / 2.0)
        
        # Grasp position: top surface + grasp_height_offset
        # grasp_height_offset is typically small (0.02m) to grasp slightly above the top surface
        grasp_pos = center_pos.copy()
        grasp_pos[2] = top_surface_z + self.grasp_height_offset
        
        rospy.loginfo(f">>> [GRASP]   Object center Z: {center_pos[2]:.3f}m")
        rospy.loginfo(f">>> [GRASP]   Top surface Z: {top_surface_z:.3f}m (center + height/2)")
        rospy.loginfo(f">>> [GRASP]   Grasp Z: {grasp_pos[2]:.3f}m (top + {self.grasp_height_offset:.3f}m)")
        
        # Top-down orientation
        roll, pitch, yaw = self.end_effector_orientation
        quat = quaternion_from_euler(roll, pitch, yaw)
        
        pose = self.arm.get_current_pose().pose
        pose.position.x = grasp_pos[0]
        pose.position.y = grasp_pos[1]
        pose.position.z = grasp_pos[2]
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        
        return pose
    
    def _open_gripper(self):
        """Open gripper."""
        rospy.loginfo(">>> [GRIPPER] Opening gripper (target position: {:.2f})...".format(self.gripper_open_position))
        self.gripper.set_joint_value_target({'hand_motor_joint': self.gripper_open_position})
        success = self.gripper.go(wait=True)
        if success:
            rospy.loginfo(">>> [GRIPPER] Gripper opened successfully")
        else:
            rospy.logwarn(">>> [GRIPPER] Gripper open command may have failed")
        rospy.sleep(0.5)
    
    def _close_gripper(self):
        """Close gripper."""
        rospy.loginfo(">>> [GRIPPER] Closing gripper (target position: {:.2f})...".format(self.gripper_close_position))
        self.gripper.set_joint_value_target({'hand_motor_joint': self.gripper_close_position})
        success = self.gripper.go(wait=True)
        if success:
            rospy.loginfo(">>> [GRIPPER] Gripper closed successfully")
        else:
            rospy.logwarn(">>> [GRIPPER] Gripper close command may have failed")
        rospy.sleep(0.5)
    
    def _check_pose_consistency(self):
        """Check if poses in buffer are consistent (similar position)."""
        if len(self.pose_buffer) < self.pose_buffer_size:
            return False
        
        # Check position consistency
        positions = []
        for pose_msg in self.pose_buffer:
            pos = np.array([
                pose_msg.pose.position.x,
                pose_msg.pose.position.y,
                pose_msg.pose.position.z
            ])
            positions.append(pos)
        
        # Calculate pairwise distances
        max_distance = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                max_distance = max(max_distance, distance)
        
        is_consistent = max_distance <= self.pose_consistency_threshold
        
        if is_consistent:
            rospy.loginfo(f"Pose buffer is consistent (max distance: {max_distance:.4f}m <= {self.pose_consistency_threshold}m)")
        else:
            rospy.loginfo_throttle(1.0, f"Pose buffer not consistent (max distance: {max_distance:.4f}m > {self.pose_consistency_threshold}m)")
        
        return is_consistent
    
    def _get_average_pose(self):
        """Get average pose from buffer."""
        if len(self.pose_buffer) == 0:
            return None
        
        # Average positions
        avg_pos = np.zeros(3)
        for pose_msg in self.pose_buffer:
            avg_pos += np.array([
                pose_msg.pose.position.x,
                pose_msg.pose.position.y,
                pose_msg.pose.position.z
            ])
        avg_pos /= len(self.pose_buffer)
        
        # Use orientation from most recent pose (or could average quaternions)
        avg_pose = PoseStamped()
        avg_pose.header = self.pose_buffer[-1].header  # Use most recent header
        avg_pose.pose.position.x = avg_pos[0]
        avg_pose.pose.position.y = avg_pos[1]
        avg_pose.pose.position.z = avg_pos[2]
        avg_pose.pose.orientation = self.pose_buffer[-1].pose.orientation  # Use most recent orientation
        
        return avg_pose
    
    def _publish_target_pose_marker(self, pose_stamped, color=(1.0, 0.0, 0.0, 1.0), scale=0.1):
        """Publish a marker for the target pose."""
        marker = Marker()
        marker.header = pose_stamped.header
        marker.ns = "target_pose"
        marker.id = self.marker_id_counter
        self.marker_id_counter += 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = pose_stamped.pose
        marker.scale.x = scale * 2.0  # Arrow length
        marker.scale.y = scale * 0.5  # Arrow width
        marker.scale.z = scale * 0.5  # Arrow height
        marker.color = ColorRGBA(*color)
        marker.lifetime = rospy.Duration(0)  # 0 = never auto-delete
        self.target_pose_marker_pub.publish(marker)
    
    def _publish_end_effector_marker(self, pose_stamped, color=(0.0, 1.0, 0.0, 1.0), scale=0.05):
        """Publish a marker for the end effector pose."""
        marker = Marker()
        marker.header = pose_stamped.header
        marker.ns = "end_effector"
        marker.id = self.marker_id_counter
        self.marker_id_counter += 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose_stamped.pose
        marker.scale.x = scale * 2.0
        marker.scale.y = scale * 2.0
        marker.scale.z = scale * 2.0
        marker.color = ColorRGBA(*color)
        marker.lifetime = rospy.Duration(0)
        self.end_effector_marker_pub.publish(marker)
    
    def _publish_path_marker(self, start_pose, end_pose_stamped, frame_id='odom', color=(0.0, 0.0, 1.0, 1.0), scale=0.02):
        """Publish a marker for the planned path (line from start to end)."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "planned_path"
        marker.id = self.marker_id_counter
        self.marker_id_counter += 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.color = ColorRGBA(*color)
        marker.lifetime = rospy.Duration(10.0)  # Show for 10 seconds
        
        # Create line from start to end
        start_pt = Point()
        start_pt.x = start_pose.position.x
        start_pt.y = start_pose.position.y
        start_pt.z = start_pose.position.z
        
        end_pt = Point()
        end_pt.x = end_pose_stamped.pose.position.x
        end_pt.y = end_pose_stamped.pose.position.y
        end_pt.z = end_pose_stamped.pose.position.z
        
        marker.points = [start_pt, end_pt]
        self.path_marker_pub.publish(marker)
    
    def _publish_status(self, status):
        """Publish current status."""
        self.status_pub.publish(String(status))
        rospy.loginfo(f"Status: {status}")

if __name__ == '__main__':
    try:
        node = PickAndPlaceNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
