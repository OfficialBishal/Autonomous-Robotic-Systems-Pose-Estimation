#!/usr/bin/env python3
"""
Helper script to generate a basic OpenCV model file from a reference image.

This script:
1. Takes an image of your object
2. Extracts ORB features
3. Creates a simplified model file (you'll need to manually add 3D coordinates)

For a complete model, use OpenCV's model registration tool from the tutorial.
"""

import cv2
import yaml
import numpy as np
import sys
import argparse

def extract_features_from_image(image_path, num_keypoints=2000):
    """
    Extract ORB features from an image.
    
    Args:
        image_path: Path to the reference image
        num_keypoints: Number of keypoints to detect
    
    Returns:
        keypoints: List of OpenCV KeyPoint objects
        descriptors: NumPy array of descriptors
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None, None
    
    # Create ORB detector
    orb = cv2.ORB.create(num_keypoints)
    
    # Detect and compute
    keypoints, descriptors = orb.detectAndCompute(img, None)
    
    print(f"Detected {len(keypoints)} keypoints")
    
    return keypoints, descriptors

def create_model_template(keypoints, descriptors, output_path):
    """
    Create a YAML model file template with extracted features.
    Note: 3D coordinates are set to placeholder values - you need to update them!
    """
    points = []
    
    for i, (kp, desc) in enumerate(zip(keypoints, descriptors)):
        # Convert descriptor to list (ORB descriptors are binary, stored as uint8)
        desc_list = desc.tolist()
        
        # Create a point entry
        # NOTE: x, y, z are placeholders! You need to:
        # 1. Define your object's coordinate frame
        # 2. Measure or estimate the 3D position of each feature point
        # 3. Update these values accordingly
        point = {
            'x': 0.0,  # TODO: Replace with actual 3D X coordinate
            'y': 0.0,  # TODO: Replace with actual 3D Y coordinate
            'z': 0.0,  # TODO: Replace with actual 3D Z coordinate
            'descriptor': desc_list,
            # Store 2D position for reference (not used by pose estimation, just for your reference)
            '_2d_x': float(kp.pt[0]),
            '_2d_y': float(kp.pt[1]),
            '_comment': f'Feature point {i} from reference image'
        }
        points.append(point)
    
    # Create YAML structure
    model_data = {
        'points': points,
        '_info': {
            'num_points': len(points),
            'descriptor_size': len(descriptors[0]) if len(descriptors) > 0 else 0,
            'note': 'This is a template. You MUST update the x, y, z coordinates with actual 3D positions!'
        }
    }
    
    # Write to file
    with open(output_path, 'w') as f:
        yaml.dump(model_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created model template at: {output_path}")
    print(f"WARNING: You need to update the x, y, z coordinates with actual 3D positions!")

def main():
    parser = argparse.ArgumentParser(
        description='Generate OpenCV model file template from a reference image'
    )
    parser.add_argument('image_path', help='Path to reference image of your object')
    parser.add_argument('-o', '--output', default='opencv_model.yml',
                       help='Output YAML file path (default: opencv_model.yml)')
    parser.add_argument('-n', '--num-keypoints', type=int, default=2000,
                       help='Number of keypoints to detect (default: 2000)')
    
    args = parser.parse_args()
    
    print("Extracting ORB features from image...")
    keypoints, descriptors = extract_features_from_image(args.image_path, args.num_keypoints)
    
    if keypoints is None:
        sys.exit(1)
    
    print("Creating model template...")
    create_model_template(keypoints, descriptors, args.output)
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Open the generated YAML file")
    print("2. For each feature point, determine its 3D position in your object's coordinate frame")
    print("3. Update the x, y, z values accordingly")
    print("4. Remove the _2d_x, _2d_y, and _comment fields (they're just for reference)")
    print("\nFor a complete automated solution, use OpenCV's model registration tool:")
    print("  https://docs.opencv.org/4.x/dc/d2c/tutorial_real_time_pose.html")

if __name__ == '__main__':
    main()

