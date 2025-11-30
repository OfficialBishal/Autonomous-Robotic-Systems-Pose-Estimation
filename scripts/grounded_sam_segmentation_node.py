#!/usr/bin/env python3
"""
Grounded SAM Segmentation ROS Node

Subscribes to RGB images, uses Grounding DINO + SAM to generate object masks,
and publishes masks for FoundationPose pose estimation.
"""

# Imports

# Standard library
import os
import sys
import numpy as np
import cv2
import torch
import torchvision

# ROS
import rospy
from sensor_msgs.msg import Image

# Add Grounded SAM paths
GROUNDED_SAM_PATH = os.path.expanduser("~/hsr_robocanes_omniverse/Grounded-Segment-Anything")
if os.path.exists(GROUNDED_SAM_PATH):
    sys.path.insert(0, GROUNDED_SAM_PATH)
    sys.path.insert(0, os.path.join(GROUNDED_SAM_PATH, "GroundingDINO"))

# Grounding DINO
try:
    import GroundingDINO.groundingdino.datasets.transforms as T
    from GroundingDINO.groundingdino.models import build_model
    from GroundingDINO.groundingdino.util.slconfig import SLConfig
    from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
except ImportError as e:
    print(f"ERROR: Failed to import Grounding DINO modules: {e}")
    print("Make sure you're running in the grounded_sam conda environment")
    sys.exit(1)

# SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError as e:
    print(f"ERROR: Failed to import SAM modules: {e}")
    print("Make sure you're running in the grounded_sam conda environment")
    sys.exit(1)

# Main Node Class

class GroundedSAMSegmentationNode:
    """
    Grounded SAM Segmentation ROS Node
    
    Subscribes to RGB camera topic, performs object detection using Grounding DINO,
    generates segmentation masks using SAM, and publishes masks for FoundationPose.
    """
    
    def __init__(self):
        """Initialize the Grounded SAM segmentation node."""
        rospy.init_node('grounded_sam_segmentation', anonymous=True)
        
        # Load parameters
        self._load_parameters()
        
        # Initialize models
        self._initialize_grounding_dino()
        self._initialize_sam()
        
        # Setup ROS communication
        self._setup_ros_communication()
        
        rospy.loginfo("Grounded SAM Segmentation node initialized")
        rospy.loginfo(f"Subscribing to RGB: {self.rgb_topic}")
        rospy.loginfo(f"Publishing mask to: {self.mask_topic}")
        rospy.loginfo(f"Text prompt: {self.text_prompt}")
    
    # Initialization Methods
    
    def _load_parameters(self):
        """Load ROS parameters from config file."""
        # Object name parameter (used for default mask topic and text prompt)
        self.object_name = rospy.get_param('~object_name', 'cracker_box')
        rospy.loginfo(f"Object name: {self.object_name}")
        
        # Topic parameters
        self.rgb_topic = rospy.get_param('~camera/rgb_topic', 
                                        rospy.get_param('~rgb_topic', '/hsrb/head_rgbd_sensor/rgb/image_rect_color'))
        self.mask_topic = rospy.get_param('~mask/mask_topic',
                                         rospy.get_param('~mask_topic', ''))
        # If mask_topic is empty, use default based on object_name
        if not self.mask_topic:
            self.mask_topic = f'/segmentation/{self.object_name}_mask'
            rospy.loginfo(f"No mask_topic specified, using default: {self.mask_topic}")
        self.frame_id = rospy.get_param('~camera/frame_id',
                                       rospy.get_param('~frame_id', 'head_rgbd_sensor_rgb_frame'))
        
        # Grounding DINO parameters
        grounded_sam_config = rospy.get_param('~grounded_sam', {})
        self.groundingdino_checkpoint = rospy.get_param('~grounded_sam/groundingdino_checkpoint',
                                                       grounded_sam_config.get('groundingdino_checkpoint',
                                                       os.path.join(GROUNDED_SAM_PATH, 'checkpoints', 'groundingdino_swint_ogc.pth')))
        self.groundingdino_config = rospy.get_param('~grounded_sam/groundingdino_config',
                                                    grounded_sam_config.get('groundingdino_config',
                                                    os.path.join(GROUNDED_SAM_PATH, 'GroundingDINO', 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py')))
        
        # Text prompt (auto-generate from object_name if empty)
        self.text_prompt = rospy.get_param('~grounded_sam/text_prompt',
                                          grounded_sam_config.get('text_prompt', ''))
        if not self.text_prompt:
            # Generate text prompt from object_name (e.g., "cracker_box" -> "cracker box")
            self.text_prompt = self.object_name.replace('_', ' ')
            rospy.loginfo(f"No text_prompt specified, using: {self.text_prompt}")
        
        # Detection thresholds
        self.box_threshold = rospy.get_param('~grounded_sam/box_threshold',
                                            grounded_sam_config.get('box_threshold', 0.3))
        self.text_threshold = rospy.get_param('~grounded_sam/text_threshold',
                                             grounded_sam_config.get('text_threshold', 0.25))
        self.iou_threshold = rospy.get_param('~grounded_sam/iou_threshold',
                                            grounded_sam_config.get('iou_threshold', 0.5))
        
        # SAM model parameters
        self.sam_model_type = rospy.get_param('~sam/sam_model_type',
                                            rospy.get_param('~sam_model_type', 'vit_h'))
        sam_checkpoint_default = os.path.join(
            os.path.expanduser('~'),
            'hsr_robocanes_omniverse',
            'segment-anything',
            'checkpoints',
            f'sam_{self.sam_model_type}.pth'
        )
        self.sam_checkpoint = rospy.get_param('~sam/sam_checkpoint',
                                             rospy.get_param('~sam_checkpoint', sam_checkpoint_default))
        
        # Device
        self.device = rospy.get_param('~device', 'cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Using device: {self.device}")
    
    def _initialize_grounding_dino(self):
        """Initialize Grounding DINO model."""
        rospy.loginfo(f"Loading Grounding DINO from {self.groundingdino_checkpoint}")
        try:
            # Expand user path
            config_path = os.path.expanduser(self.groundingdino_config)
            checkpoint_path = os.path.expanduser(self.groundingdino_checkpoint)
            
            if not os.path.exists(config_path):
                rospy.logerr(f"Grounding DINO config not found: {config_path}")
                sys.exit(1)
            if not os.path.exists(checkpoint_path):
                rospy.logerr(f"Grounding DINO checkpoint not found: {checkpoint_path}")
                sys.exit(1)
            
            # Load model
            args = SLConfig.fromfile(config_path)
            args.device = self.device
            self.grounding_dino_model = build_model(args)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            load_res = self.grounding_dino_model.load_state_dict(
                clean_state_dict(checkpoint["model"]), strict=False
            )
            rospy.loginfo(f"Grounding DINO load result: {load_res}")
            self.grounding_dino_model = self.grounding_dino_model.to(self.device)
            self.grounding_dino_model.eval()
            
            # Image transform for Grounding DINO
            self.grounding_transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            rospy.loginfo("Grounding DINO model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to initialize Grounding DINO: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            sys.exit(1)
    
    def _initialize_sam(self):
        """Initialize SAM model."""
        rospy.loginfo(f"Loading SAM model: {self.sam_model_type} from {self.sam_checkpoint}")
        try:
            checkpoint_path = os.path.expanduser(self.sam_checkpoint)
            if not os.path.exists(checkpoint_path):
                rospy.logerr(f"SAM checkpoint not found: {checkpoint_path}")
                sys.exit(1)
            
            # Load SAM model
            self.sam = sam_model_registry[self.sam_model_type](checkpoint=checkpoint_path)
            self.sam.to(device=self.device)
            self.sam_predictor = SamPredictor(self.sam)
            
            rospy.loginfo("SAM model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to initialize SAM: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            sys.exit(1)
    
    def _setup_ros_communication(self):
        """Setup ROS publishers and subscribers."""
        # Publisher for mask
        self.mask_pub = rospy.Publisher(self.mask_topic, Image, queue_size=10)
        
        # Subscriber for RGB image
        self.image_sub = rospy.Subscriber(
            self.rgb_topic,
            Image,
            self.image_callback,
            queue_size=1
        )
        
        rospy.loginfo(f"Subscribed to: {self.rgb_topic}")
        rospy.loginfo(f"Publishing to: {self.mask_topic}")
    
    # Image Processing
    
    def ros_image_to_numpy(self, img_msg, desired_encoding='rgb8'):
        """
        Convert ROS Image message to numpy array.
        
        Args:
            img_msg: sensor_msgs.msg.Image
            desired_encoding: Desired output encoding (e.g., 'rgb8', 'bgr8')
        
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
        else:
            rospy.logwarn(f"Unsupported encoding: {img_msg.encoding}, attempting default conversion")
            if len(img_msg.data) == height * width:
                img_array = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(height, width)
            elif len(img_msg.data) == height * width * 3:
                img_array = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(height, width, 3)
            else:
                raise ValueError(f"Cannot convert encoding {img_msg.encoding} to {desired_encoding}")
        
        return img_array
    
    def numpy_to_ros_image(self, img_array, encoding='mono8', frame_id=None):
        """
        Convert numpy array to ROS Image message.
        
        Args:
            img_array: numpy.ndarray image
            encoding: ROS image encoding (e.g., 'mono8', 'rgb8')
            frame_id: Frame ID for header
        
        Returns:
            sensor_msgs.msg.Image: ROS Image message
        """
        img_msg = Image()
        img_msg.header.stamp = rospy.Time.now()
        if frame_id:
            img_msg.header.frame_id = frame_id
        
        if len(img_array.shape) == 2:
            # Grayscale
            height, width = img_array.shape
            img_msg.height = height
            img_msg.width = width
            img_msg.encoding = encoding
            img_msg.is_bigendian = 0
            img_msg.step = width
            img_msg.data = img_array.tobytes()
        elif len(img_array.shape) == 3:
            # Color
            height, width, channels = img_array.shape
            img_msg.height = height
            img_msg.width = width
            img_msg.encoding = encoding
            img_msg.is_bigendian = 0
            img_msg.step = width * channels
            img_msg.data = img_array.tobytes()
        else:
            raise ValueError(f"Unsupported image shape: {img_array.shape}")
        
        return img_msg
    
    # Grounding DINO Detection
    
    def get_grounding_output(self, image, caption, box_threshold, text_threshold):
        """
        Get object detection output from Grounding DINO.
        
        Args:
            image: PIL Image
            caption: Text prompt (e.g., "cracker box")
            box_threshold: Box confidence threshold
            text_threshold: Text confidence threshold
        
        Returns:
            tuple: (boxes_filt, scores, pred_phrases)
        """
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        
        # Transform image
        image_tensor, _ = self.grounding_transform(image, None)
        
        # Move to device
        self.grounding_dino_model = self.grounding_dino_model.to(self.device)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_dino_model(image_tensor[None], captions=[caption])
        
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        
        # Filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        
        # Get phrases
        tokenizer = self.grounding_dino_model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            pred_phrases.append(pred_phrase)
            scores.append(logit.max().item())
        
        return boxes_filt, torch.Tensor(scores), pred_phrases
    
    # Segmentation Methods
    
    def segment_with_grounded_sam(self, image_rgb):
        """
        Segment object using Grounding DINO + SAM.
        
        Args:
            image_rgb: RGB image as numpy array
        
        Returns:
            numpy.ndarray: Binary mask, or None if no object detected
        """
        try:
            # Convert numpy array to PIL Image
            from PIL import Image
            image_pil = Image.fromarray(image_rgb)
            
            # Get bounding boxes from Grounding DINO
            boxes_filt, scores, pred_phrases = self.get_grounding_output(
                image_pil, self.text_prompt, self.box_threshold, self.text_threshold
            )
            
            if boxes_filt.size(0) == 0:
                rospy.logwarn_throttle(5.0, f"No objects detected for prompt: {self.text_prompt}")
                return None
            
            rospy.loginfo_throttle(5.0, f"Detected {boxes_filt.size(0)} objects: {pred_phrases}")
            
            # Convert boxes from normalized [cx, cy, w, h] to [x1, y1, x2, y2] in image coordinates
            H, W = image_rgb.shape[:2]
            boxes_xyxy = boxes_filt.clone()
            for i in range(boxes_xyxy.size(0)):
                boxes_xyxy[i] = boxes_xyxy[i] * torch.Tensor([W, H, W, H])
                boxes_xyxy[i][:2] -= boxes_xyxy[i][2:] / 2
                boxes_xyxy[i][2:] += boxes_xyxy[i][:2]
            
            boxes_xyxy = boxes_xyxy.cpu()
            
            # Apply NMS to remove overlapping boxes
            if boxes_xyxy.size(0) > 1:
                nms_idx = torchvision.ops.nms(boxes_xyxy, scores, self.iou_threshold).numpy().tolist()
                boxes_xyxy = boxes_xyxy[nms_idx]
                scores = scores[nms_idx]
                pred_phrases = [pred_phrases[idx] for idx in nms_idx]
                rospy.loginfo_throttle(5.0, f"After NMS: {boxes_xyxy.size(0)} boxes")
            
            # Use the highest confidence box
            best_idx = scores.argmax().item()
            best_box = boxes_xyxy[best_idx].numpy()
            
            # Set image for SAM predictor
            self.sam_predictor.set_image(image_rgb)
            
            # Transform box for SAM
            transformed_box = self.sam_predictor.transform.apply_boxes_torch(
                torch.from_numpy(best_box[None, :]).to(self.device),
                image_rgb.shape[:2]
            )
            
            # Generate mask
            masks, scores_sam, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_box,
                multimask_output=False,
            )
            
            # Get the mask (first and only one)
            mask = masks[0, 0].cpu().numpy().astype(np.uint8)
            
            # Convert to binary mask (0 or 255)
            binary_mask = (mask * 255).astype(np.uint8)
            
            rospy.loginfo_throttle(5.0, f"Generated mask for: {pred_phrases[best_idx]}")
            
            return binary_mask
            
        except Exception as e:
            rospy.logerr(f"Error in segment_with_grounded_sam: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return None
    
    # ROS Callbacks
    
    def image_callback(self, img_msg):
        """Callback for RGB image messages."""
        try:
            # Convert ROS image to numpy
            image_rgb = self.ros_image_to_numpy(img_msg, desired_encoding='rgb8')
            
            # Segment object
            mask = self.segment_with_grounded_sam(image_rgb)
            
            if mask is None:
                # Publish empty mask
                empty_mask = np.zeros((img_msg.height, img_msg.width), dtype=np.uint8)
                mask_msg = self.numpy_to_ros_image(empty_mask, encoding='mono8', frame_id=self.frame_id)
                self.mask_pub.publish(mask_msg)
                return
            
            # Publish mask
            mask_msg = self.numpy_to_ros_image(mask, encoding='mono8', frame_id=self.frame_id)
            self.mask_pub.publish(mask_msg)
            
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())


def main():
    """Main function."""
    try:
        node = GroundedSAMSegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error in Grounded SAM node: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())


if __name__ == '__main__':
    main()

