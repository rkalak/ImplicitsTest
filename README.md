# Implicit CAD Kernel

A Python library for working with Signed Distance Functions (SDFs) and performing topology optimization. This project provides a complete toolkit for implicit CAD modeling, allowing you to create complex 3D shapes using mathematical functions rather than explicit geometry.

## Features

- **SDF Primitives**: Basic geometric shapes (sphere, box, cylinder, torus, plane)
- **CSG Operations**: Union, intersection, difference, and smooth blending
- **Transformations**: Translation, rotation, scaling, and arbitrary matrix transforms
- **Topology Optimization**: Level-set method for structural optimization
- **Mesh Generation**: Marching cubes algorithm for converting SDFs to triangle meshes
- **Export Formats**: STL and OBJ export support

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from implicit_cad import Sphere, Box, Union, Difference, MeshGenerator

# Create primitives
sphere = Sphere(radius=1.0, center=(0, 0, 0))
box = Box(size=(1.5, 1.5, 1.5), center=(0, 0, 0))

# Combine using CSG operations
result = Difference(sphere, box)

# Generate mesh
bounds = ((-3, -3, -3), (3, 3, 3))
resolution = (64, 64, 64)
generator = MeshGenerator(bounds, resolution)
mesh = generator.generate_mesh(result)

# Export to STL
generator.export_stl(result, "output.stl")
```

### Topology Optimization

```python
from implicit_cad import TopologyOptimizer, MeshGenerator
import numpy as np

def compliance_objective(density_field):
    # Define your objective function
    # Returns: (objective_value, sensitivity_field)
    objective = np.sum(density_field)
    sensitivity = -np.ones_like(density_field)
    return objective, sensitivity

# Create optimizer
bounds = ((0, 0, 0), (4, 1, 1))
resolution = (80, 20, 20)
optimizer = TopologyOptimizer(
    bounds=bounds,
    resolution=resolution,
    volume_fraction=0.3,
    filter_radius=1.5,
)

# Run optimization
final_phi = optimizer.evolve_level_set(
    objective_func=compliance_objective,
    max_iterations=50,
)

# Generate mesh from optimized result
optimized_sdf = optimizer.get_sdf_from_level_set()
generator = MeshGenerator(bounds, resolution)
mesh = generator.generate_mesh(optimized_sdf)
generator.export_stl(optimized_sdf, "optimized.stl")
```

## Architecture

### Core Components

1. **SDF Base Class** (`sdf.py`): Abstract base class for all signed distance functions
2. **Primitives** (`primitives.py`): Basic geometric shapes
3. **Operations** (`operations.py`): CSG operations for combining SDFs
4. **Transforms** (`transforms.py`): Geometric transformations
5. **Topology Optimizer** (`topology_optimizer.py`): Level-set based optimization
6. **Mesh Generator** (`mesh_generator.py`): Marching cubes mesh generation

### SDF Concepts

A Signed Distance Function (SDF) is a function that returns the signed distance from any point in 3D space to the surface of a shape:
- **Negative values**: Point is inside the shape
- **Positive values**: Point is outside the shape
- **Zero**: Point is on the surface

SDFs enable powerful operations:
- **CSG Operations**: Combine shapes using min/max operations
- **Smooth Blending**: Create organic transitions between shapes
- **Topology Optimization**: Evolve shapes to optimize objectives

## Examples

See the `examples/` directory for complete examples:

- `basic_primitives.py`: Creating and combining basic shapes
- `smooth_blending.py`: Using smooth union and intersection
- `topology_optimization.py`: Structural optimization example
- `visualize_sdf.py`: Visualizing SDF values and isosurfaces

Run examples:
```bash
python examples/basic_primitives.py
python examples/topology_optimization.py
```

## Topology Optimization

The topology optimizer uses a level-set method to evolve a shape and minimize an objective function (e.g., compliance) while satisfying constraints (e.g., volume fraction).

### How It Works

1. **Level-Set Representation**: The shape is represented as a level-set function (SDF)
2. **Density Field**: Converted to a density field using a smooth Heaviside function
3. **Objective Evaluation**: Compute objective and sensitivity (gradient)
4. **Sensitivity Filtering**: Apply filtering to reduce mesh dependency
5. **Level-Set Evolution**: Update the level-set function using gradient descent
6. **Reinitialization**: Periodically reinitialize to maintain signed distance property

### Custom Objectives

To use topology optimization, provide an objective function that:
- Takes a density field (numpy array) as input
- Returns `(objective_value, sensitivity_field)` tuple
- The sensitivity field indicates where to add/remove material

Example:
```python
def my_objective(density_field):
    # Compute objective (e.g., compliance, stress, etc.)
    objective = compute_compliance(density_field)
    
    # Compute sensitivity (gradient w.r.t. density)
    sensitivity = compute_sensitivity(density_field)
    
    return objective, sensitivity
```

## Advanced Usage

### Custom SDFs

Create your own SDF by subclassing the `SDF` base class:

```python
from implicit_cad import SDF

class MyCustomSDF(SDF):
    def distance(self, point):
        x, y, z = point
        # Your distance function here
        return np.sqrt(x**2 + y**2 + z**2) - 1.0
```

### Combining Multiple Shapes

```python
from implicit_cad import Sphere, Union, Translate

# Create multiple spheres
spheres = [Translate(Sphere(radius=0.5), (i, 0, 0)) 
           for i in range(5)]

# Combine them
result = spheres[0]
for s in spheres[1:]:
    result = Union(result, s)
```

### Smooth Blending

```python
from implicit_cad import Sphere, SmoothUnion

s1 = Sphere(radius=1.0, center=(-0.5, 0, 0))
s2 = Sphere(radius=1.0, center=(0.5, 0, 0))

# Smooth union with blending factor k
blended = SmoothUnion(s1, s2, k=0.3)  # Larger k = smoother blend
```

## Performance Tips

- Use appropriate grid resolution: Higher resolution = better quality but slower
- For topology optimization, start with lower resolution and refine
- Use smooth operations sparingly as they're more expensive
- Consider using numpy vectorization for custom SDFs

## Dependencies

- `numpy`: Numerical computations
- `scipy`: Scientific computing and optimization
- `scikit-image`: Marching cubes algorithm
- `trimesh`: Mesh manipulation
- `matplotlib`: Visualization (optional)
- `pyvista`: 3D visualization (optional)

## References

- [Signed Distance Functions](https://iquilezles.org/articles/distfunctions/)
- [Level-Set Methods for Structural Topology Optimization](https://www.researchgate.net/publication/228636456_Level_set_methods_for_structural_topology_optimization)
- [Marching Cubes Algorithm](https://en.wikipedia.org/wiki/Marching_cubes)

## License

This project is for educational purposes. Feel free to use and modify as needed.

## Contributing

This is an educational project. Suggestions and improvements are welcome!
