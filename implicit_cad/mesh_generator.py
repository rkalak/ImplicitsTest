"""
Mesh generation from SDFs using marching cubes algorithm.
"""

import numpy as np
from typing import Tuple, Optional
import trimesh
from .sdf import SDF, Point3D


class MeshGenerator:
    """Generate triangle meshes from SDFs using marching cubes."""
    
    def __init__(self, bounds: Tuple[Point3D, Point3D], resolution: Tuple[int, int, int]):
        """
        Args:
            bounds: ((x_min, y_min, z_min), (x_max, y_max, z_max))
            resolution: (nx, ny, nz) grid resolution for marching cubes
        """
        self.bounds = bounds
        self.resolution = resolution
    
    def generate_mesh(self, sdf: SDF, iso_value: float = 0.0) -> trimesh.Trimesh:
        """
        Generate a triangle mesh from an SDF using marching cubes.
        
        Args:
            sdf: The SDF to mesh
            iso_value: Iso-surface value (typically 0.0 for zero level set)
            
        Returns:
            Trimesh object
        """
        from skimage import measure
        
        # Evaluate SDF on grid
        sdf_grid = sdf.evaluate_grid(self.bounds, self.resolution)
        
        # Apply marching cubes
        vertices, faces, normals, values = measure.marching_cubes(
            sdf_grid,
            level=iso_value,
            spacing=self._compute_spacing(),
            allow_degenerate=False,
        )
        
        # Transform vertices to world coordinates
        (x_min, y_min, z_min), (x_max, y_max, z_max) = self.bounds
        nx, ny, nz = self.resolution
        
        # Marching cubes returns vertices in grid coordinates, need to transform
        vertices[:, 0] = x_min + vertices[:, 0] * (x_max - x_min) / (nx - 1)
        vertices[:, 1] = y_min + vertices[:, 1] * (y_max - y_min) / (ny - 1)
        vertices[:, 2] = z_min + vertices[:, 2] * (z_max - z_min) / (nz - 1)
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        
        # Remove degenerate faces and fix normals
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        
        return mesh
    
    def _compute_spacing(self) -> Tuple[float, float, float]:
        """Compute spacing for marching cubes."""
        (x_min, y_min, z_min), (x_max, y_max, z_max) = self.bounds
        nx, ny, nz = self.resolution
        
        dx = (x_max - x_min) / (nx - 1) if nx > 1 else 1.0
        dy = (y_max - y_min) / (ny - 1) if ny > 1 else 1.0
        dz = (z_max - z_min) / (nz - 1) if nz > 1 else 1.0
        
        return (dx, dy, dz)
    
    def export_stl(self, sdf: SDF, filename: str, iso_value: float = 0.0) -> None:
        """
        Export SDF as STL file.
        
        Args:
            sdf: The SDF to export
            filename: Output filename (.stl)
            iso_value: Iso-surface value
        """
        mesh = self.generate_mesh(sdf, iso_value)
        mesh.export(filename)
    
    def export_obj(self, sdf: SDF, filename: str, iso_value: float = 0.0) -> None:
        """
        Export SDF as OBJ file.
        
        Args:
            sdf: The SDF to export
            filename: Output filename (.obj)
            iso_value: Iso-surface value
        """
        mesh = self.generate_mesh(sdf, iso_value)
        mesh.export(filename)
