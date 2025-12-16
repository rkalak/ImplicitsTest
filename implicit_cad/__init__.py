"""
Implicit CAD Kernel - A library for working with Signed Distance Functions (SDFs)
and performing topology optimization.
"""

from .sdf import SDF, Point3D
from .primitives import Sphere, Box, Cylinder, Torus, Plane
from .operations import Union, Intersection, Difference, SmoothUnion, SmoothIntersection
from .transforms import Translate, Rotate, Scale, Transform
from .topology_optimizer import TopologyOptimizer
from .mesh_generator import MeshGenerator

__version__ = "0.1.0"

__all__ = [
    "SDF",
    "Point3D",
    "Sphere",
    "Box",
    "Cylinder",
    "Torus",
    "Plane",
    "Union",
    "Intersection",
    "Difference",
    "SmoothUnion",
    "SmoothIntersection",
    "Translate",
    "Rotate",
    "Scale",
    "Transform",
    "TopologyOptimizer",
    "MeshGenerator",
]
