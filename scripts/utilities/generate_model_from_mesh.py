#!/usr/bin/env python3
"""
Generate OpenCV model file from existing 3D mesh and reference image.

This script:
1. Loads a 3D mesh file (PLY format)
2. Takes a reference image of the object
3. Extracts ORB features from the image
4. Projects 3D mesh points to 2D using camera pose
5. Creates the OpenCV model YAML file

Usage:
    python generate_model_from_mesh.py --mesh path/to/mesh.ply --image path/to/image.jpg --output model.yml
"""

import cv2
import numpy as np
import yaml
import argparse
import sys
import os

def load_ply_mesh(ply_path):
    """
    Load a PLY mesh file and extract vertices.
    Returns vertices as Nx3 numpy array.
    """
    vertices = []
    try:
        with open(ply_path, 'r') as f:
            lines = f.readlines()
            
        # Find header
        vertex_count = 0
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            if line.startswith('end_header'):
                header_end = i + 1
                break
        
        # Read vertices
        for i in range(header_end, header_end + vertex_count):
            parts = lines[i].split()
            if len(parts) >= 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                vertices.append([x, y, z])
        
        return np.array(vertices)
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        return None

def extract_features_from_image(image_path, num_keypoints=2000):
    """Extract ORB features from an image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None, None
    
    orb = cv2.ORB.create(num_keypoints)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    
    print(f"Extracted {len(keypoints)} keypoints from image")
    return keypoints, descriptors, img

def project_3d_to_2d(vertices_3d, camera_matrix, rvec, tvec, dist_coeffs=None):
    """Project 3D points to 2D image coordinates."""
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))
    
    points_2d, _ = cv2.projectPoints(vertices_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)

def find_correspondences(keypoints, descriptors, mesh_vertices_2d, mesh_vertices_3d, max_distance=10.0):
    """
    Find correspondences between image keypoints and mesh vertices.
    For each keypoint, find the nearest mesh vertex in 2D space.
    """
    correspondences = []
    
    for i, kp in enumerate(keypoints):
        kp_pos = np.array([kp.pt[0], kp.pt[1]])
        
        # Find nearest mesh vertex in 2D
        distances = np.linalg.norm(mesh_vertices_2d - kp_pos, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_distance = distances[nearest_idx]
        
        if nearest_distance < max_distance:
            correspondences.append({
                'keypoint_idx': i,
                'mesh_vertex_idx': nearest_idx,
                'keypoint': kp,
                'descriptor': descriptors[i],
                '3d_point': mesh_vertices_3d[nearest_idx],
                'distance': nearest_distance
            })
    
    print(f"Found {len(correspondences)} correspondences")
    return correspondences

def create_model_file(correspondences, output_path):
    """Create OpenCV model YAML file from correspondences."""
    points = []
    
    for corr in correspondences:
        pt_3d = corr['3d_point']
        desc = corr['descriptor']
        
        # Convert descriptor to list (ORB descriptors are binary, 0-255)
        desc_list = desc.tolist()
        
        point = {
            'x': float(pt_3d[0]),
            'y': float(pt_3d[1]),
            'z': float(pt_3d[2]),
            'descriptor': desc_list
        }
        points.append(point)
    
    model_data = {
        'points': points,
        '_info': {
            'num_points': len(points),
            'descriptor_size': len(desc_list) if points else 0,
            'generated_from': 'mesh_and_image'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(model_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created model file with {len(points)} points: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate OpenCV model from 3D mesh and reference image'
    )
    parser.add_argument('--mesh', required=True, help='Path to PLY mesh file')
    parser.add_argument('--image', required=True, help='Path to reference image')
    parser.add_argument('--output', default='opencv_model.yml', help='Output YAML file')
    parser.add_argument('--num-keypoints', type=int, default=2000, help='Number of ORB keypoints')
    parser.add_argument('--max-distance', type=float, default=10.0, 
                       help='Max distance for keypoint-vertex matching (pixels)')
    
    # Camera parameters (can be estimated or provided)
    parser.add_argument('--fx', type=float, default=525.0, help='Camera focal length X')
    parser.add_argument('--fy', type=float, default=525.0, help='Camera focal length Y')
    parser.add_argument('--cx', type=float, default=320.0, help='Camera principal point X')
    parser.add_argument('--cy', type=float, default=240.0, help='Camera principal point Y')
    
    args = parser.parse_args()
    
    # Load mesh
    print(f"Loading mesh from {args.mesh}...")
    mesh_vertices = load_ply_mesh(args.mesh)
    if mesh_vertices is None:
        sys.exit(1)
    
    print(f"Loaded {len(mesh_vertices)} vertices from mesh")
    
    # Normalize mesh to reasonable scale (PLY files might be in mm, convert to meters)
    # Assume mesh is centered at origin, scale if needed
    mesh_center = np.mean(mesh_vertices, axis=0)
    mesh_vertices_centered = mesh_vertices - mesh_center
    
    # Scale to meters (assuming PLY might be in cm or mm)
    max_dim = np.max(np.abs(mesh_vertices_centered))
    if max_dim > 1.0:  # Likely in cm or mm
        scale = 0.01 if max_dim > 100 else 0.001
        mesh_vertices_scaled = mesh_vertices_centered * scale
        print(f"Scaled mesh by {scale} (assuming original units were {'cm' if max_dim > 100 else 'mm'})")
    else:
        mesh_vertices_scaled = mesh_vertices_centered
    
    # Extract features from image
    print(f"Extracting features from {args.image}...")
    keypoints, descriptors, img = extract_features_from_image(args.image, args.num_keypoints)
    if keypoints is None:
        sys.exit(1)
    
    # Create camera matrix
    camera_matrix = np.array([
        [args.fx, 0, args.cx],
        [0, args.fy, args.cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Estimate camera pose (simplified - assumes object is centered and facing camera)
    # In practice, you'd use PnP with known correspondences or manual alignment
    img_height, img_width = img.shape[:2]
    
    # Simple pose estimation: object centered, at reasonable distance
    # This is a placeholder - for real use, you'd need proper camera pose
    tvec = np.array([[0], [0], [0.5]], dtype=np.float32)  # 50cm away
    rvec = np.array([[0], [0], [0]], dtype=np.float32)  # No rotation
    
    print("Projecting 3D mesh to 2D...")
    mesh_vertices_2d = project_3d_to_2d(mesh_vertices_scaled, camera_matrix, rvec, tvec)
    
    # Find correspondences
    print("Finding correspondences...")
    correspondences = find_correspondences(keypoints, descriptors, mesh_vertices_2d, 
                                          mesh_vertices_scaled, args.max_distance)
    
    if len(correspondences) < 4:
        print(f"Warning: Only {len(correspondences)} correspondences found. Need at least 4 for PnP.")
        print("You may need to:")
        print("  1. Adjust --max-distance parameter")
        print("  2. Ensure the image shows the object clearly")
        print("  3. Provide better camera pose estimation")
    
    # Create model file
    create_model_file(correspondences, args.output)
    
    print("\n" + "="*60)
    print("Model file created successfully!")
    print("="*60)
    print(f"Output: {args.output}")
    print(f"Points: {len(correspondences)}")
    print("\nNote: Camera pose estimation is simplified. For better results:")
    print("  - Use a calibrated camera")
    print("  - Provide accurate camera pose")
    print("  - Or use OpenCV's model registration tool from the tutorial")

if __name__ == '__main__':
    main()

