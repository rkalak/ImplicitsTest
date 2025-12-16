"""
Visualize SDF values as a 3D volume rendering.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from implicit_cad import Sphere, Box, Union, Difference

def visualize_sdf_2d_slice(sdf, bounds, resolution, z_slice=0.0):
    """
    Visualize a 2D slice of the SDF.
    
    Args:
        sdf: The SDF to visualize
        bounds: Bounds of the domain
        resolution: Grid resolution
        z_slice: Z coordinate of the slice
    """
    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounds
    nx, ny, _ = resolution
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate SDF on slice
    sdf_values = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            sdf_values[i, j] = sdf.distance((X[i, j], Y[i, j], z_slice))
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, sdf_values, levels=50, cmap='RdYlBu')
    plt.colorbar(label='Signed Distance')
    plt.contour(X, Y, sdf_values, levels=[0], colors='black', linewidths=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'SDF Slice at Z = {z_slice}')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_sdf_3d_isosurface(sdf, bounds, resolution):
    """
    Visualize 3D isosurface of the SDF.
    
    Args:
        sdf: The SDF to visualize
        bounds: Bounds of the domain
        resolution: Grid resolution
    """
    from implicit_cad import MeshGenerator
    
    generator = MeshGenerator(bounds, resolution)
    mesh = generator.generate_mesh(sdf)
    
    # Use pyvista for 3D visualization if available
    try:
        import pyvista as pv
        
        # Convert trimesh to pyvista
        pv_mesh = pv.wrap(mesh)
        
        # Create plotter
        plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh, color='lightblue', show_edges=True)
        plotter.show()
    except ImportError:
        print("PyVista not available. Using matplotlib 3D plot instead...")
        
        # Fallback to matplotlib
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample points on mesh surface
        vertices = mesh.vertices
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  c='blue', s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('SDF Isosurface')
        plt.show()

def main():
    # Create a complex shape
    sphere1 = Sphere(radius=1.0, center=(-0.5, 0, 0))
    sphere2 = Sphere(radius=1.0, center=(0.5, 0, 0))
    box = Box(size=(0.8, 0.8, 2.0), center=(0, 0, 0))
    
    # Combine
    union = Union(sphere1, sphere2)
    result = Difference(union, box)
    
    # Visualize 2D slice
    bounds = ((-3, -3, -3), (3, 3, 3))
    resolution = (100, 100, 100)
    
    print("Visualizing 2D slice...")
    visualize_sdf_2d_slice(result, bounds, resolution, z_slice=0.0)
    
    # Visualize 3D isosurface
    print("Visualizing 3D isosurface...")
    visualize_sdf_3d_isosurface(result, bounds, resolution)

if __name__ == "__main__":
    main()
