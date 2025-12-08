# Development Tools

These scripts are standalone development and testing tools. They are **not required** for the main pose estimation pipeline and are not used by any ROS nodes or launch files.

## Scripts

- `analyze_mesh_simple.py` - Analyze mesh file properties (OBJ format)
- `generate_model_from_image.py` - Generate 3D model from image
- `generate_model_from_mesh.py` - Generate model from mesh file
- `generate_mustard_model.py` - Generate mustard bottle model

## Usage

These tools can be run manually for development/testing purposes:

```bash
# Analyze a mesh file
python tools/analyze_mesh_simple.py path/to/mesh.obj

# Generate model from mesh and image
python tools/generate_model_from_mesh.py --mesh mesh.ply --image image.jpg --output model.yml
```

**Note**: These scripts are kept for reference but are not part of the main system. They were used during development but are not called by any ROS nodes or launch files.

