"""
Topology optimization using level-set methods and sensitivity analysis.
"""

import numpy as np
from typing import Callable, Tuple, Optional
from .sdf import SDF, Point3D


class TopologyOptimizer:
    """
    Topology optimizer using level-set method with SDFs.
    
    This optimizer evolves a level-set function (represented as an SDF) to minimize
    an objective function (e.g., compliance) while satisfying constraints (e.g., volume).
    """
    
    def __init__(
        self,
        bounds: Tuple[Point3D, Point3D],
        resolution: Tuple[int, int, int],
        volume_fraction: float = 0.5,
        filter_radius: float = 1.5,
        move_limit: float = 0.2,
    ):
        """
        Args:
            bounds: ((x_min, y_min, z_min), (x_max, y_max, z_max))
            resolution: (nx, ny, nz) grid resolution
            volume_fraction: Target volume fraction (0-1)
            filter_radius: Radius for sensitivity filtering
            move_limit: Maximum change per iteration
        """
        self.bounds = bounds
        self.resolution = resolution
        self.volume_fraction = volume_fraction
        self.filter_radius = filter_radius
        self.move_limit = move_limit
        
        # Create grid
        (x_min, y_min, z_min), (x_max, y_max, z_max) = bounds
        nx, ny, nz = resolution
        
        self.x = np.linspace(x_min, x_max, nx)
        self.y = np.linspace(y_min, y_max, ny)
        self.z = np.linspace(z_min, z_max, nz)
        
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.grid_points = np.stack([self.X.flatten(), self.Y.flatten(), self.Z.flatten()], axis=1)
        
        # Initialize level-set function (SDF values)
        self.phi = np.ones((nx, ny, nz))  # Start with all material (negative = material)
        
        # Compute grid spacing
        self.dx = (x_max - x_min) / (nx - 1) if nx > 1 else 1.0
        self.dy = (y_max - y_min) / (ny - 1) if ny > 1 else 1.0
        self.dz = (z_max - z_min) / (nz - 1) if nz > 1 else 1.0
    
    def compute_density(self, phi: np.ndarray) -> np.ndarray:
        """
        Convert level-set function to density using Heaviside approximation.
        
        Args:
            phi: Level-set function (SDF values)
            
        Returns:
            Density field (0 = void, 1 = material)
        """
        # Smooth Heaviside function
        eta = 0.5 * self.dx  # Smoothing parameter
        rho = 1.0 / (1.0 + np.exp(-phi / eta))
        return rho
    
    def filter_sensitivity(self, sensitivity: np.ndarray) -> np.ndarray:
        """
        Apply sensitivity filtering to reduce mesh dependency.
        
        Args:
            sensitivity: Raw sensitivity field
            
        Returns:
            Filtered sensitivity field
        """
        from scipy.ndimage import gaussian_filter
        # Convert filter radius to grid units
        sigma = self.filter_radius / min(self.dx, self.dy, self.dz)
        return gaussian_filter(sensitivity, sigma=sigma)
    
    def compute_volume(self, phi: np.ndarray) -> float:
        """Compute volume fraction from level-set function."""
        rho = self.compute_density(phi)
        return np.mean(rho)
    
    def evolve_level_set(
        self,
        objective_func: Callable[[np.ndarray], Tuple[float, np.ndarray]],
        max_iterations: int = 50,
        convergence_tol: float = 1e-4,
        callback: Optional[Callable[[int, float, float], None]] = None,
    ) -> np.ndarray:
        """
        Evolve the level-set function to optimize the objective.
        
        Args:
            objective_func: Function that takes density field and returns (objective, sensitivity)
            max_iterations: Maximum number of iterations
            convergence_tol: Convergence tolerance
            callback: Optional callback(iteration, objective, volume)
            
        Returns:
            Final level-set function (SDF)
        """
        for iteration in range(max_iterations):
            # Compute density from level-set
            rho = self.compute_density(self.phi)
            
            # Evaluate objective and compute sensitivity
            objective, sensitivity = objective_func(rho.reshape(self.resolution))
            
            # Filter sensitivity
            sensitivity_filtered = self.filter_sensitivity(sensitivity)
            
            # Compute volume constraint sensitivity
            volume = self.compute_volume(self.phi)
            volume_sensitivity = -np.ones_like(self.phi)  # Negative = remove material
            
            # Combine sensitivities (Lagrange multiplier for volume constraint)
            # Simple approach: use volume error as multiplier
            volume_error = volume - self.volume_fraction
            lambda_vol = 1000.0 * volume_error  # Adaptive multiplier
            
            total_sensitivity = sensitivity_filtered + lambda_vol * volume_sensitivity
            
            # Normalize sensitivity
            if np.max(np.abs(total_sensitivity)) > 0:
                total_sensitivity = total_sensitivity / np.max(np.abs(total_sensitivity))
            
            # Update level-set function (gradient descent)
            dt = self.move_limit * min(self.dx, self.dy, self.dz)
            self.phi = self.phi - dt * total_sensitivity
            
            # Reinitialize level-set periodically to maintain signed distance property
            if iteration % 10 == 0:
                self.phi = self.reinitialize_level_set(self.phi)
            
            # Callback
            if callback:
                callback(iteration, objective, volume)
            
            # Check convergence
            if iteration > 0 and abs(volume_error) < convergence_tol:
                break
        
        return self.phi
    
    def reinitialize_level_set(self, phi: np.ndarray) -> np.ndarray:
        """
        Reinitialize level-set function to maintain signed distance property.
        Uses fast marching method approximation.
        """
        # Simple reinitialization: compute distance to zero level set
        from scipy.ndimage import distance_transform_edt
        
        # Create binary mask
        mask = phi < 0
        
        # Compute distance to boundary
        dist_inside = distance_transform_edt(mask)
        dist_outside = distance_transform_edt(~mask)
        
        # Combine with sign
        phi_reinit = dist_outside - dist_inside
        
        return phi_reinit
    
    def get_sdf_from_level_set(self) -> 'LevelSetSDF':
        """
        Convert the optimized level-set function to an SDF object.
        
        Returns:
            LevelSetSDF object that can be used like any other SDF
        """
        return LevelSetSDF(self.phi, self.bounds, self.resolution)


class LevelSetSDF(SDF):
    """SDF created from a level-set function on a grid."""
    
    def __init__(self, phi: np.ndarray, bounds: Tuple[Point3D, Point3D], resolution: Tuple[int, int, int]):
        """
        Args:
            phi: Level-set function values on grid
            bounds: Bounds of the grid
            resolution: Grid resolution
        """
        self.phi = phi
        self.bounds = bounds
        self.resolution = resolution
        
        (x_min, y_min, z_min), (x_max, y_max, z_max) = bounds
        nx, ny, nz = resolution
        
        self.x = np.linspace(x_min, x_max, nx)
        self.y = np.linspace(y_min, y_max, ny)
        self.z = np.linspace(z_min, z_max, nz)
    
    def distance(self, point: Point3D) -> float:
        """Interpolate SDF value at point using trilinear interpolation."""
        from scipy.interpolate import RegularGridInterpolator
        
        p = np.array(point, dtype=float)
        
        # Clamp to bounds
        (x_min, y_min, z_min), (x_max, y_max, z_max) = self.bounds
        p[0] = np.clip(p[0], x_min, x_max)
        p[1] = np.clip(p[1], y_min, y_max)
        p[2] = np.clip(p[2], z_min, z_max)
        
        # Interpolate
        interp = RegularGridInterpolator(
            (self.x, self.y, self.z),
            self.phi,
            method='linear',
            bounds_error=False,
            fill_value=np.max(self.phi)  # Outside bounds = outside material
        )
        
        return float(interp(p))
