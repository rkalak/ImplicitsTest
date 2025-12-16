"""
Example demonstrating smooth blending operations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from implicit_cad import Sphere, SmoothUnion, SmoothIntersection, Translate, MeshGenerator

def main():
    # Create two spheres
    sphere1 = Sphere(radius=1.0, center=(-0.5, 0, 0))
    sphere2 = Sphere(radius=1.0, center=(0.5, 0, 0))
    
    # Smooth union with different blending factors
    smooth_union = SmoothUnion(sphere1, sphere2, k=0.3)
    
    # Create another pair for smooth intersection
    sphere3 = Sphere(radius=0.8, center=(0, -0.5, 0))
    sphere4 = Sphere(radius=0.8, center=(0, 0.5, 0))
    smooth_intersection = SmoothIntersection(sphere3, sphere4, k=0.2)
    
    # Combine both
    result = SmoothUnion(smooth_union, smooth_intersection, k=0.4)
    
    # Generate mesh
    bounds = ((-3, -3, -3), (3, 3, 3))
    resolution = (64, 64, 64)
    generator = MeshGenerator(bounds, resolution)
    
    print("Generating mesh with smooth blending...")
    mesh = generator.generate_mesh(result)
    print(f"Mesh generated: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Export
    output_file = "smooth_blending.stl"
    generator.export_stl(result, output_file)
    print(f"Exported to {output_file}")

if __name__ == "__main__":
    main()
