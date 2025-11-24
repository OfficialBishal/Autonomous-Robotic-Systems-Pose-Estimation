#!/usr/bin/env python3
"""
Simple mesh orientation analysis by parsing OBJ file directly.
"""

import sys
import os
import re

def analyze_obj_file(obj_file):
    """Analyze OBJ file to determine mesh orientation."""
    print("="*70)
    print("MESH ORIENTATION ANALYSIS (from OBJ file)")
    print("="*70)
    
    vertices = []
    
    print(f"\nParsing OBJ file: {obj_file}")
    with open(obj_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v ') and not line.startswith('vn'):  # Vertex line
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        vertices.append([x, y, z])
                    except ValueError:
                        continue
    
    if not vertices:
        print("Error: No vertices found in OBJ file")
        return
    
    vertices = [[v[0], v[1], v[2]] for v in vertices]
    print(f"Found {len(vertices)} vertices")
    
    # Calculate bounds
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    
    extent_x = max_x - min_x
    extent_y = max_y - min_y
    extent_z = max_z - min_z
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    
    print(f"\n--- Mesh Bounds ---")
    print(f"X-axis: min={min_x:.6f}, max={max_x:.6f}, extent={extent_x:.6f}")
    print(f"Y-axis: min={min_y:.6f}, max={max_y:.6f}, extent={extent_y:.6f}")
    print(f"Z-axis: min={min_z:.6f}, max={max_z:.6f}, extent={extent_z:.6f}")
    print(f"\nCenter: [{center_x:.6f}, {center_y:.6f}, {center_z:.6f}]")
    
    # Determine which axis is "up" (height) - largest extent
    extents = [extent_x, extent_y, extent_z]
    max_extent = max(extents)
    extent_ratios = [e / max_extent for e in extents]
    
    print(f"\n--- Extent Analysis ---")
    print(f"Extent ratios (normalized): X={extent_ratios[0]:.3f}, Y={extent_ratios[1]:.3f}, Z={extent_ratios[2]:.3f}")
    
    sorted_indices = sorted(range(3), key=lambda i: extents[i], reverse=True)
    axis_names = ['X', 'Y', 'Z']
    
    print(f"\n--- Axis Ranking (by extent) ---")
    for i, idx in enumerate(sorted_indices):
        print(f"  {i+1}. {axis_names[idx]}-axis: {extents[idx]:.6f} ({extent_ratios[idx]:.3f} of max)")
    
    # For a mustard bottle (cylindrical), the height should be the largest
    height_axis = sorted_indices[0]
    
    print(f"\n--- Interpretation ---")
    print(f"Assuming this is a mustard bottle (cylindrical object):")
    print(f"  Height axis (up): {axis_names[height_axis]}")
    print(f"  Width axis:       {axis_names[sorted_indices[1]]}")
    print(f"  Depth axis:       {axis_names[sorted_indices[2]]}")
    
    # Check centering
    center_mag = (center_x**2 + center_y**2 + center_z**2)**0.5
    print(f"\n--- Centering Check ---")
    if center_mag < 0.01:
        print(f"Mesh is centered (center magnitude: {center_mag:.6f})")
    else:
        print(f"✗ Mesh is NOT centered (center magnitude: {center_mag:.6f})")
        print(f"  Note: FoundationPose will automatically center it")
    
    # Determine coordinate convention
    print(f"\n--- Coordinate Convention ---")
    print(f"Common conventions:")
    print(f"  - Blender/OBJ: Usually +Y up or +Z up")
    print(f"  - OpenGL: +Y up")
    print(f"  - ROS: +Z up")
    print(f"  - FoundationPose: Uses mesh as-is")
    
    if height_axis == 1:  # Y is height
        print(f"\nMesh appears to use Y-up convention (common in Blender/OBJ)")
    elif height_axis == 2:  # Z is height
        print(f"\nMesh appears to use Z-up convention (common in ROS)")
    else:  # X is height (unusual)
        print(f"\nMesh appears to use X-up convention (unusual)")
    
    # Sample some vertices to understand orientation
    print(f"\n--- Sample Vertices (first 5) ---")
    for i, v in enumerate(vertices[:5]):
        print(f"  Vertex {i+1}: [{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}]")
    
    print(f"\n--- Recommendations ---")
    print(f"1. Check the mesh in a 3D viewer (MeshLab, Blender) to confirm orientation")
    print(f"2. If estimated pose axes don't match desired object frame:")
    print(f"   - Use ~mesh_coordinate_correction to pre-transform the mesh")
    print(f"   - OR use ~coordinate_correction to post-transform the pose")
    print(f"3. FoundationPose camera frame: X right, Y down, Z forward (OpenCV)")
    
    return {
        'extents': extents,
        'height_axis': height_axis,
        'center': [center_x, center_y, center_z]
    }

if __name__ == '__main__':
    default_mesh = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'meshes', 'mustard.obj'
    )
    
    mesh_file = sys.argv[1] if len(sys.argv) > 1 else default_mesh
    
    if not os.path.exists(mesh_file):
        print(f"Error: Mesh file not found: {mesh_file}")
        sys.exit(1)
    
    try:
        result = analyze_obj_file(mesh_file)
        print("\n" + "="*70)
        print("Analysis complete!")
        print("="*70)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

