#!/usr/bin/env python3
"""
Create an OpenCV FileStorage format model file from 3D points and descriptors.

This script creates a model file compatible with OpenCV's Model::load() function.
The file format must be OpenCV FileStorage YAML with:
- points_3d: cv::Mat (Nx1, 3 channels, float)
- descriptors: cv::Mat (Nx32 for ORB)
- keypoints: (optional) vector of KeyPoint
- training_image_path: (optional) string
"""

import cv2
import numpy as np
import sys
import argparse

def create_opencv_model_file(points_3d, descriptors, keypoints=None, training_image_path="", output_path="model.yml"):
    """
    Create an OpenCV FileStorage format model file.
    
    Args:
        points_3d: List of 3D points [(x, y, z), ...] or numpy array (N, 3)
        descriptors: numpy array of descriptors (N, descriptor_size)
        keypoints: (optional) List of cv2.KeyPoint objects
        training_image_path: (optional) Path to training image
        output_path: Output file path
    """
    # Convert to numpy arrays
    if isinstance(points_3d, list):
        points_3d = np.array(points_3d, dtype=np.float32)
    
    if not isinstance(descriptors, np.ndarray):
        descriptors = np.array(descriptors, dtype=np.uint8)
    
    # Ensure points_3d is Nx1x3 format (OpenCV expects this)
    if points_3d.ndim == 2 and points_3d.shape[1] == 3:
        # Reshape to Nx1x3
        points_3d = points_3d.reshape(-1, 1, 3)
    
    # Ensure descriptors are uint8 (ORB descriptors are binary)
    if descriptors.dtype != np.uint8:
        descriptors = descriptors.astype(np.uint8)
    
    # Create FileStorage
    fs = cv2.FileStorage(output_path, cv2.FileStorage_WRITE)
    
    # Write points_3d
    fs.write("points_3d", points_3d)
    
    # Write descriptors
    fs.write("descriptors", descriptors)
    
    # Write keypoints if provided
    if keypoints is not None:
        fs.write("keypoints", keypoints)
    
    # Write training image path if provided
    if training_image_path:
        fs.write("training_image_path", training_image_path)
    
    fs.release()
    
    print(f"Created OpenCV model file: {output_path}")
    print(f"  Points: {len(points_3d)}")
    print(f"  Descriptors: {descriptors.shape}")
    if keypoints:
        print(f"  Keypoints: {len(keypoints)}")

def create_simple_mustard_model(output_path):
    """
    Create a simple model file for mustard bottle with placeholder descriptors.
    This is just for testing - real tracking requires descriptors from a reference image.
    """
    # Mustard bottle dimensions: 9.6cm x 19.1cm x 5.8cm (from DOPE config)
    # Create 8 corner points of a bounding box (in meters, centered at origin)
    width = 0.096 / 2.0   # half width
    height = 0.191 / 2.0  # half height
    depth = 0.058 / 2.0   # half depth
    
    points_3d = [
        [width, height, depth],      # Front Top Right
        [-width, height, depth],     # Front Top Left
        [-width, -height, depth],   # Front Bottom Left
        [width, -height, depth],     # Front Bottom Right
        [width, height, -depth],     # Rear Top Right
        [-width, height, -depth],    # Rear Top Left
        [-width, -height, -depth],  # Rear Bottom Left
        [width, -height, -depth],   # Rear Bottom Right
    ]
    
    # Create placeholder ORB descriptors (32 bytes each, 8 descriptors)
    # These are just placeholders - real descriptors should come from a reference image
    num_points = len(points_3d)
    descriptors = np.random.randint(0, 256, (num_points, 32), dtype=np.uint8)
    
    # Create keypoints (optional, but helpful)
    keypoints = []
    for i, pt in enumerate(points_3d):
        kp = cv2.KeyPoint()
        kp.pt = (0, 0)  # 2D position not known without reference image
        kp.size = 10.0
        kp.angle = -1
        kp.response = 1.0
        kp.octave = 0
        kp.class_id = i
        keypoints.append(kp)
    
    create_opencv_model_file(points_3d, descriptors, keypoints, "", output_path)
    print("\nWARNING: This model uses placeholder descriptors!")
    print("For real tracking, you need to:")
    print("1. Take a reference image of the mustard bottle")
    print("2. Extract ORB features from that image")
    print("3. Match features to 3D mesh points")
    print("4. Create a proper model file with real descriptors")

def main():
    parser = argparse.ArgumentParser(description='Create OpenCV FileStorage format model file')
    parser.add_argument('--output', '-o', default='opencv_model_mustard.yml',
                       help='Output file path')
    parser.add_argument('--simple', action='store_true',
                       help='Create a simple test model with placeholder descriptors')
    
    args = parser.parse_args()
    
    if args.simple:
        create_simple_mustard_model(args.output)
    else:
        print("Use --simple to create a test model, or provide points and descriptors")
        print("For a complete model, use OpenCV's model registration tool")

if __name__ == '__main__':
    main()


