"""
Topology optimization example: Optimize a cantilever beam.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from implicit_cad import TopologyOptimizer, MeshGenerator

def compliance_objective(density_field):
    """
    Simplified compliance objective for demonstration.
    In a real implementation, this would solve the elasticity equations.
    
    Args:
        density_field: Density field (0 = void, 1 = material)
        
    Returns:
        (objective_value, sensitivity_field)
    """
    # Simplified: minimize volume while keeping some structure
    # In practice, you'd solve: K * u = f, compliance = u^T * K * u
    
    # For demonstration, use a simple gradient-based objective
    # that encourages material in certain regions
    
    nx, ny, nz = density_field.shape
    
    # Create a "load" pattern (e.g., force at one end)
    load_pattern = np.zeros_like(density_field)
    load_pattern[-1, :, :] = 1.0  # Force at one end
    
    # Simplified compliance: material should be where loads are
    # and along paths connecting loads to supports
    objective = np.sum(density_field * load_pattern)
    
    # Sensitivity: negative where we want material (minimize compliance)
    sensitivity = -load_pattern.copy()
    
    # Add connectivity term (encourage material between load and support)
    support_pattern = np.zeros_like(density_field)
    support_pattern[0, :, :] = 1.0  # Support at other end
    
    # Gradient-based connectivity
    for i in range(1, nx):
        connectivity = np.exp(-i / (nx * 0.3))  # Decay from support
        sensitivity[i, :, :] -= connectivity * 0.5
    
    return objective, sensitivity

def main():
    # Define optimization domain
    bounds = ((0, 0, 0), (4, 1, 1))
    resolution = (80, 20, 20)  # Higher resolution for better results
    
    # Create optimizer
    optimizer = TopologyOptimizer(
        bounds=bounds,
        resolution=resolution,
        volume_fraction=0.3,  # Use 30% of material
        filter_radius=1.5,
        move_limit=0.2,
    )
    
    # Track optimization history
    history = []
    
    def callback(iteration, objective, volume):
        history.append((iteration, objective, volume))
        if iteration % 5 == 0:
            print(f"Iteration {iteration}: Objective = {objective:.6f}, Volume = {volume:.4f}")
    
    print("Starting topology optimization...")
    print("This may take a few minutes...")
    
    # Run optimization
    final_phi = optimizer.evolve_level_set(
        objective_func=compliance_objective,
        max_iterations=50,
        convergence_tol=1e-4,
        callback=callback,
    )
    
    print(f"\nOptimization complete!")
    print(f"Final volume fraction: {optimizer.compute_volume(final_phi):.4f}")
    
    # Convert to SDF and generate mesh
    print("\nGenerating mesh...")
    optimized_sdf = optimizer.get_sdf_from_level_set()
    
    mesh_generator = MeshGenerator(bounds, resolution)
    mesh = mesh_generator.generate_mesh(optimized_sdf)
    print(f"Mesh generated: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Export
    output_file = "topology_optimized.stl"
    mesh_generator.export_stl(optimized_sdf, output_file)
    print(f"Exported to {output_file}")
    
    # Plot optimization history
    try:
        import matplotlib.pyplot as plt
        
        iterations = [h[0] for h in history]
        objectives = [h[1] for h in history]
        volumes = [h[2] for h in history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(iterations, objectives)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Objective')
        ax1.set_title('Objective History')
        ax1.grid(True)
        
        ax2.plot(iterations, volumes, label='Volume')
        ax2.axhline(y=optimizer.volume_fraction, color='r', linestyle='--', label='Target')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Volume Fraction')
        ax2.set_title('Volume History')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('optimization_history.png')
        print("Saved optimization history plot to optimization_history.png")
    except ImportError:
        print("Matplotlib not available, skipping plot")

if __name__ == "__main__":
    main()
