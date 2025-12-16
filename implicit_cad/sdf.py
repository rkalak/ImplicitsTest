"""
Core SDF (Signed Distance Function) base class and utilities.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np

Point3D = Union[Tuple[float, float, float], np.ndarray]


class SDF(ABC):
    """
    Base class for all Signed Distance Functions.
    
    A Signed Distance Function returns the signed distance from a point to the surface
    of a shape. Negative values indicate the point is inside, positive values indicate
    outside, and zero indicates on the surface.
    """
    
    @abstractmethod
    def distance(self, point: Point3D) -> float:
        """
        Compute the signed distance from a point to the surface.
        
        Args:
            point: 3D point (x, y, z)
            
        Returns:
            Signed distance (negative = inside, positive = outside, zero = on surface)
        """
        pass
    
    def __call__(self, point: Point3D) -> float:
        """Allow SDF objects to be called directly."""
        return self.distance(point)
    
    def evaluate_grid(self, bounds: Tuple[Point3D, Point3D], resolution: Tuple[int, int, int]) -> np.ndarray:
        """
        Evaluate the SDF on a regular grid.
        
        Args:
            bounds: ((x_min, y_min, z_min), (x_max, y_max, z_max))
            resolution: (nx, ny, nz) number of points in each dimension
            
        Returns:
            3D array of signed distance values
        """
        (x_min, y_min, z_min), (x_max, y_max, z_max) = bounds
        nx, ny, nz = resolution
        
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        z = np.linspace(z_min, z_max, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        distances = np.array([self.distance(p) for p in points])
        return distances.reshape((nx, ny, nz))
    
    def contains(self, point: Point3D) -> bool:
        """Check if a point is inside the shape (distance < 0)."""
        return self.distance(point) < 0
    
    def union(self, other: 'SDF') -> 'Union':
        """Create a union of this SDF with another."""
        from .operations import Union
        return Union(self, other)
    
    def intersection(self, other: 'SDF') -> 'Intersection':
        """Create an intersection of this SDF with another."""
        from .operations import Intersection
        return Intersection(self, other)
    
    def difference(self, other: 'SDF') -> 'Difference':
        """Create a difference (subtraction) of this SDF from another."""
        from .operations import Difference
        return Difference(self, other)
