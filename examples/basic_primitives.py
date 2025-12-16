"""
Basic example: Creating and combining primitive shapes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from implicit_cad import Sphere, Box, Cylinder, Union, Difference, Translate, Rotate
from implicit_cad import MeshGenerator

def main():
    # Create some primitives
    sphere = Sphere(radius=1.0, center=(0, 0, 0))
    box = Box(size=(1.5, 1.5, 1.5), center=(0, 0, 0))
    
    # Create a cylinder
    cylinder = Cylinder(radius=0.5, height=2.0, center=(0, 0, 0))
    
    # Combine shapes using CSG operations
    # Union: combine sphere and box
    combined = Union(sphere, box)
    
    # Difference: subtract cylinder from combined shape
    result = Difference(combined, cylinder)
    
    # Rotate the result
    rotated = Rotate(result, axis=(1, 1, 0), angle=0.5)
    
    # Generate mesh
    bounds = ((-3, -3, -3), (3, 3, 3))
    resolution = (64, 64, 64)
    generator = MeshGenerator(bounds, resolution)
    
    print("Generating mesh...")
    mesh = generator.generate_mesh(rotated)
    print(f"Mesh generated: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Export to STL
    output_file = "basic_primitives.stl"
    generator.export_stl(rotated, output_file)
    print(f"Exported to {output_file}")

if __name__ == "__main__":
    main()
