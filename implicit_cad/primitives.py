"""
Basic geometric primitives implemented as SDFs.
"""

import numpy as np
from typing import Tuple
from .sdf import SDF, Point3D


class Sphere(SDF):
    """A sphere primitive."""
    
    def __init__(self, radius: float = 1.0, center: Point3D = (0, 0, 0)):
        """
        Args:
            radius: Radius of the sphere
            center: Center point (x, y, z)
        """
        self.radius = radius
        self.center = np.array(center, dtype=float)
    
    def distance(self, point: Point3D) -> float:
        p = np.array(point, dtype=float)
        return np.linalg.norm(p - self.center) - self.radius


class Box(SDF):
    """A box (rectangular prism) primitive."""
    
    def __init__(self, size: Point3D = (1, 1, 1), center: Point3D = (0, 0, 0)):
        """
        Args:
            size: Size of the box (width, height, depth)
            center: Center point (x, y, z)
        """
        self.size = np.array(size, dtype=float)
        self.center = np.array(center, dtype=float)
        self.half_size = self.size / 2.0
    
    def distance(self, point: Point3D) -> float:
        p = np.array(point, dtype=float) - self.center
        q = np.abs(p) - self.half_size
        return np.linalg.norm(np.maximum(q, 0)) + min(np.max(q), 0)


class Cylinder(SDF):
    """A cylinder primitive (axis-aligned along z)."""
    
    def __init__(self, radius: float = 1.0, height: float = 2.0, center: Point3D = (0, 0, 0)):
        """
        Args:
            radius: Radius of the cylinder
            height: Height of the cylinder
            center: Center point (x, y, z)
        """
        self.radius = radius
        self.height = height
        self.center = np.array(center, dtype=float)
    
    def distance(self, point: Point3D) -> float:
        p = np.array(point, dtype=float) - self.center
        d = np.array([np.linalg.norm(p[:2]), abs(p[2]) - self.height / 2.0])
        return min(np.max(d), 0) + np.linalg.norm(np.maximum(d, 0)) - self.radius


class Torus(SDF):
    """A torus primitive."""
    
    def __init__(self, major_radius: float = 1.0, minor_radius: float = 0.3, center: Point3D = (0, 0, 0)):
        """
        Args:
            major_radius: Major radius (distance from center to tube center)
            minor_radius: Minor radius (radius of the tube)
            center: Center point (x, y, z)
        """
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.center = np.array(center, dtype=float)
    
    def distance(self, point: Point3D) -> float:
        p = np.array(point, dtype=float) - self.center
        q = np.array([np.linalg.norm(p[:2]) - self.major_radius, p[2]])
        return np.linalg.norm(q) - self.minor_radius


class Plane(SDF):
    """An infinite plane primitive."""
    
    def __init__(self, normal: Point3D = (0, 0, 1), point: Point3D = (0, 0, 0)):
        """
        Args:
            normal: Normal vector of the plane (will be normalized)
            point: A point on the plane
        """
        self.normal = np.array(normal, dtype=float)
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.point = np.array(point, dtype=float)
    
    def distance(self, point: Point3D) -> float:
        p = np.array(point, dtype=float)
        return np.dot(p - self.point, self.normal)
