"""
CSG (Constructive Solid Geometry) operations for combining SDFs.
"""

import numpy as np
from .sdf import SDF, Point3D


class Union(SDF):
    """Union of two SDFs (combines shapes)."""
    
    def __init__(self, a: SDF, b: SDF):
        self.a = a
        self.b = b
    
    def distance(self, point: Point3D) -> float:
        return min(self.a.distance(point), self.b.distance(point))


class Intersection(SDF):
    """Intersection of two SDFs (overlapping region)."""
    
    def __init__(self, a: SDF, b: SDF):
        self.a = a
        self.b = b
    
    def distance(self, point: Point3D) -> float:
        return max(self.a.distance(point), self.b.distance(point))


class Difference(SDF):
    """Difference of two SDFs (subtract b from a)."""
    
    def __init__(self, a: SDF, b: SDF):
        self.a = a
        self.b = b
    
    def distance(self, point: Point3D) -> float:
        return max(self.a.distance(point), -self.b.distance(point))


class SmoothUnion(SDF):
    """Smooth union with controllable blending."""
    
    def __init__(self, a: SDF, b: SDF, k: float = 0.1):
        """
        Args:
            a: First SDF
            b: Second SDF
            k: Smoothing factor (larger = smoother blend)
        """
        self.a = a
        self.b = b
        self.k = k
    
    def distance(self, point: Point3D) -> float:
        d1 = self.a.distance(point)
        d2 = self.b.distance(point)
        h = np.clip(0.5 + 0.5 * (d2 - d1) / self.k, 0.0, 1.0)
        return d1 * (1 - h) + d2 * h - self.k * h * (1 - h)


class SmoothIntersection(SDF):
    """Smooth intersection with controllable blending."""
    
    def __init__(self, a: SDF, b: SDF, k: float = 0.1):
        """
        Args:
            a: First SDF
            b: Second SDF
            k: Smoothing factor (larger = smoother blend)
        """
        self.a = a
        self.b = b
        self.k = k
    
    def distance(self, point: Point3D) -> float:
        d1 = self.a.distance(point)
        d2 = self.b.distance(point)
        h = np.clip(0.5 + 0.5 * (d2 - d1) / self.k, 0.0, 1.0)
        return d1 * (1 - h) + d2 * h + self.k * h * (1 - h)
