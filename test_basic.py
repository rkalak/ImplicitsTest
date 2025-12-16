"""
Basic test to verify the implicit CAD kernel works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from implicit_cad import Sphere, Box, Union, Difference, MeshGenerator

def test_basic_operations():
    """Test basic SDF operations."""
    print("Testing basic SDF operations...")
    
    # Create primitives
    sphere = Sphere(radius=1.0)
    box = Box(size=(1.5, 1.5, 1.5))
    
    # Test distance evaluation
    assert sphere.distance((0, 0, 0)) < 0, "Center should be inside sphere"
    assert sphere.distance((2, 0, 0)) > 0, "Point outside should have positive distance"
    
    # Test CSG operations
    union = Union(sphere, box)
    diff = Difference(sphere, box)
    
    # Test that operations work
    d1 = union.distance((0, 0, 0))
    d2 = diff.distance((0, 0, 0))
    
    print(f"  ✓ Sphere center distance: {sphere.distance((0, 0, 0)):.3f}")
    print(f"  ✓ Union distance at origin: {d1:.3f}")
    print(f"  ✓ Difference distance at origin: {d2:.3f}")
    
    print("Basic operations test passed!")

def test_mesh_generation():
    """Test mesh generation."""
    print("\nTesting mesh generation...")
    
    sphere = Sphere(radius=1.0)
    bounds = ((-2, -2, -2), (2, 2, 2))
    resolution = (32, 32, 32)  # Lower resolution for faster test
    
    generator = MeshGenerator(bounds, resolution)
    mesh = generator.generate_mesh(sphere)
    
    assert len(mesh.vertices) > 0, "Mesh should have vertices"
    assert len(mesh.faces) > 0, "Mesh should have faces"
    
    print(f"  ✓ Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print("Mesh generation test passed!")

if __name__ == "__main__":
    try:
        test_basic_operations()
        test_mesh_generation()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
