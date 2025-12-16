"""
Transformations for SDFs (translation, rotation, scaling).
"""

import numpy as np
from .sdf import SDF, Point3D


class Translate(SDF):
    """Translate an SDF."""
    
    def __init__(self, sdf: SDF, offset: Point3D):
        """
        Args:
            sdf: The SDF to translate
            offset: Translation vector (x, y, z)
        """
        self.sdf = sdf
        self.offset = np.array(offset, dtype=float)
    
    def distance(self, point: Point3D) -> float:
        p = np.array(point, dtype=float) - self.offset
        return self.sdf.distance(p)


class Rotate(SDF):
    """Rotate an SDF around an axis."""
    
    def __init__(self, sdf: SDF, axis: Point3D, angle: float):
        """
        Args:
            sdf: The SDF to rotate
            axis: Rotation axis vector (will be normalized)
            angle: Rotation angle in radians
        """
        self.sdf = sdf
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        
        # Rodrigues' rotation formula
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        self.rotation_matrix = (np.eye(3) * cos_a + 
                                sin_a * K + 
                                (1 - cos_a) * np.outer(axis, axis))
        self.inverse_rotation = self.rotation_matrix.T
    
    def distance(self, point: Point3D) -> float:
        p = np.array(point, dtype=float)
        p_rotated = self.inverse_rotation @ p
        return self.sdf.distance(p_rotated)


class Scale(SDF):
    """Scale an SDF."""
    
    def __init__(self, sdf: SDF, factor: float):
        """
        Args:
            sdf: The SDF to scale
            factor: Scaling factor (uniform scaling)
        """
        self.sdf = sdf
        self.factor = factor
    
    def distance(self, point: Point3D) -> float:
        p = np.array(point, dtype=float) / self.factor
        return self.sdf.distance(p) * self.factor


class Transform(SDF):
    """Apply an arbitrary 4x4 transformation matrix."""
    
    def __init__(self, sdf: SDF, matrix: np.ndarray):
        """
        Args:
            sdf: The SDF to transform
            matrix: 4x4 transformation matrix (homogeneous coordinates)
        """
        self.sdf = sdf
        self.matrix = np.array(matrix, dtype=float)
        if self.matrix.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")
        
        # Extract rotation/scale and translation
        self.rotation_scale = self.matrix[:3, :3]
        self.translation = self.matrix[:3, 3]
        
        # For inverse transform
        self.inv_rotation_scale = np.linalg.inv(self.rotation_scale)
    
    def distance(self, point: Point3D) -> float:
        p = np.array(point, dtype=float)
        # Apply inverse transform
        p_transformed = self.inv_rotation_scale @ (p - self.translation)
        # Scale distance by the average scale factor
        scale_factor = np.cbrt(np.linalg.det(self.rotation_scale))
        return self.sdf.distance(p_transformed) * scale_factor
