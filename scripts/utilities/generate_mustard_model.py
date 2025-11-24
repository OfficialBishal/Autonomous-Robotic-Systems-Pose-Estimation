#!/usr/bin/env python3
"""
Automatically generate OpenCV model file for mustard bottle using existing mesh.

This uses the YCB dataset mesh that's already in the workspace.
"""

import cv2
import numpy as np
import yaml
import sys
import os

# Paths
WS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MESH_PATH = os.path.join(WS_ROOT, 'src/usd/ycb/006_mustard_bottle/google_16k/nontextured.ply')
OUTPUT_PATH = os.path.join(WS_ROOT, 'src/final-project-OfficialBishal/config/opencv_model_mustard.yml')

# Mustard dimensions from DOPE config (in cm, converted to meters)
MUSTARD_DIMS = [0.096024150848388672, 0.19130100250244141, 0.05824894905090332]  # [width, height, depth] in meters

def load_ply_vertices(ply_path):
    """Load vertices from PLY file."""
    vertices = []
    try:
        with open(ply_path, 'r') as f:
            lines = f.readlines()
        
        vertex_count = 0
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            if line.startswith('end_header'):
                header_end = i + 1
                break
        
        for i in range(header_end, header_end + vertex_count):
            parts = lines[i].split()
            if len(parts) >= 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                vertices.append([x, y, z])
        
        return np.array(vertices)
    except Exception as e:
        print(f"Error loading PLY: {e}")
        return None

def sample_mesh_points(vertices, num_points=100):
    """Sample points from mesh vertices."""
    if len(vertices) <= num_points:
        return vertices
    
    # Random sampling
    indices = np.random.choice(len(vertices), num_points, replace=False)
    return vertices[indices]

def generate_orb_descriptors_for_points(points_3d, num_descriptors=32):
    """
    Generate synthetic ORB-like descriptors for 3D points.
    In practice, these would come from a reference image, but this allows
    the system to run. The descriptors won't match real images, but the
    structure is correct.
    """
    descriptors = []
    for i, pt in enumerate(points_3d):
        # Create a descriptor based on point position (simplified)
        # Real descriptors would come from image features
        desc = np.zeros(32, dtype=np.uint8)
        # Use point coordinates to seed descriptor
        seed = int((abs(pt[0]) + abs(pt[1]) + abs(pt[2])) * 1000) % 256
        np.random.seed(seed)
        desc = np.random.randint(0, 256, 32, dtype=np.uint8)
        descriptors.append(desc)
    return descriptors

def create_model_file(points_3d, descriptors, output_path):
    """Create OpenCV model YAML file."""
    points = []
    
    for pt, desc in zip(points_3d, descriptors):
        point = {
            'x': float(pt[0]),
            'y': float(pt[1]),
            'z': float(pt[2]),
            'descriptor': desc.tolist()
        }
        points.append(point)
    
    model_data = {
        'points': points,
        '_info': {
            'num_points': len(points),
            'descriptor_size': 32,
            'object': 'mustard_bottle',
            'mesh_source': 'YCB_dataset',
            'note': 'Descriptors are synthetic. For real tracking, extract from reference image.'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(model_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created model file: {output_path}")
    print(f"  Points: {len(points)}")
    print(f"  Object: Mustard Bottle")
    print(f"  Note: Descriptors are synthetic placeholders")

def main():
    print("Generating OpenCV model for mustard bottle...")
    print(f"Mesh path: {MESH_PATH}")
    
    if not os.path.exists(MESH_PATH):
        print(f"Error: Mesh file not found at {MESH_PATH}")
        print("Please ensure the YCB dataset is available in the workspace.")
        sys.exit(1)
    
    # Load mesh
    print("Loading mesh vertices...")
    vertices = load_ply_vertices(MESH_PATH)
    if vertices is None:
        sys.exit(1)
    
    print(f"Loaded {len(vertices)} vertices")
    
    # Normalize mesh (PLY files are typically in mm, convert to meters)
    # Center the mesh
    mesh_center = np.mean(vertices, axis=0)
    vertices_centered = vertices - mesh_center
    
    # Scale to meters (assuming mm)
    max_dim = np.max(np.abs(vertices_centered))
    if max_dim > 1.0:
        # Likely in mm, convert to meters
        scale = 0.001
        vertices_scaled = vertices_centered * scale
        print(f"Scaled mesh by {scale} (mm to meters)")
    else:
        vertices_scaled = vertices_centered
    
    # Sample points (use a reasonable number for matching)
    print("Sampling mesh points...")
    sampled_points = sample_mesh_points(vertices_scaled, num_points=200)
    
    # Generate synthetic descriptors
    print("Generating descriptors...")
    descriptors = generate_orb_descriptors_for_points(sampled_points)
    
    # Create model file
    print("Creating model file...")
    create_model_file(sampled_points, descriptors, OUTPUT_PATH)
    
    print("\n" + "="*60)
    print("Model file created!")
    print("="*60)
    print(f"Location: {OUTPUT_PATH}")
    print("\nNote: This model has synthetic descriptors.")
    print("For real tracking, you need to:")
    print("  1. Take a reference image of the mustard bottle")
    print("  2. Extract ORB features from that image")
    print("  3. Match features to 3D mesh points")
    print("  4. Update the model file with real descriptors")
    print("\nOr use the generate_model_from_mesh.py script with a reference image.")

if __name__ == '__main__':
    main()

