import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from simulation_assembly import MolecularDynamicsSimulation
import matplotlib.pyplot as plt
import networkx as nx
from typing import Literal
import os

class Ellipsoid:
    def __init__(self, positions, topology, scaling=1):
        self.positions = positions*scaling
        self.topology = topology
        self.scaling = scaling
        
        node_ids = sorted(topology.nodes())
        self.node_id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        self.fig = None
        self.ax = None
        
        self.x = None
        self.y = None
        self.z = None
        
        self.center = None
        self.radii = None
        self.rotation_matrix = None

    def combine_topology_position(self):
        G = nx.Graph()
        for node_id, pos_idx in self.node_id_map.items():
            G.add_node(node_id, pos=np.array(self.positions[pos_idx, :]))
            
        G.add_edges_from(self.topology.edges(data=True))
        
        return G
        
    def triangle_area(self, p1, p2, p3):
        return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        
    def calc_surface_area(self, mode: Literal['triangulation', 'fit'], print=False):
        topology = self.combine_topology_position()

        if mode == 'triangulation':
            surface_area_triangulation = 0
            for u, v, w in nx.find_cliques(topology):  # Find triangles (cliques of size 3)
                p1, p2, p3 = topology.nodes[u]['pos'], topology.nodes[v]['pos'], topology.nodes[w]['pos']
                surface_area_triangulation += self.triangle_area(p1, p2, p3)
            
            if print:
                print(f'Surface area (Triangulation): {surface_area_triangulation}')
            return surface_area_triangulation
        
        elif mode == 'fit':
            a = self.radii[0]
            b = self.radii[1]
            c = self.radii[2]
            p = 1.6075
            term = ((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3
            surface_area_fit = 4 * np.pi * (term ** (1 / p))
            
            if print:
                print(f'Surface Area (Fit): {surface_area_fit}')
            return surface_area_fit
   
    def fit_ellipsoid_pca(self):
        # Ensure positions is a NumPy array
        positions = np.asarray(self.positions)

        # Center the data
        self.center = positions.mean(axis=0)
        positions_centered = positions - self.center

        # Compute covariance matrix
        cov_matrix = np.cov(positions_centered, rowvar=False)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.rotation_matrix = eigenvectors

        # Project data onto principal axes
        projected_data = positions_centered @ eigenvectors

        # Compute the ranges along each principal axis
        min_proj = projected_data.min(axis=0)
        max_proj = projected_data.max(axis=0)
        ranges = max_proj - min_proj
        self.radii = ranges / 2.0  # Radii are half the ranges

    def draw_positions(self, colors=None):
        self.x, self.y, self.z = self.positions[:, 0], self.positions[:, 1], self.positions[:, 2]
        self.ax.scatter(self.x, self.y, self.z, c=colors if colors else 'blue', label='Nodes')

    def draw_ellipsoid(self, resolution=50):
        # Generate data for "unrotated" ellipsoid
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = self.radii[0] * np.outer(np.cos(u), np.sin(v))
        y = self.radii[1] * np.outer(np.sin(u), np.sin(v))
        z = self.radii[2] * np.outer(np.ones_like(u), np.cos(v))

        # Flatten the arrays and apply rotation
        xyz = np.array([x.flatten(), y.flatten(), z.flatten()])
        rotated_xyz = self.rotation_matrix @ xyz
        x_rot = rotated_xyz[0, :].reshape((resolution, resolution)) + self.center[0]
        y_rot = rotated_xyz[1, :].reshape((resolution, resolution)) + self.center[1]
        z_rot = rotated_xyz[2, :].reshape((resolution, resolution)) + self.center[2]

        # Plot the surface
        self.ax.plot_surface(x_rot, y_rot, z_rot, rstride=4, cstride=4, color='red', alpha=0.2, edgecolor='none', label='Fitted Ellipsoid Surface')

    def plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.draw_positions()
        self.draw_ellipsoid()
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        margin = 0
        
        x_min, x_max = min(self.x) - margin, max(self.x) + margin
        y_min, y_max = min(self.y) - margin, max(self.y) + margin
        z_min, z_max = min(self.z) - margin, max(self.z) + margin
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        self.ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
        self.ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
        self.ax.set_zlim(z_center - max_range / 2, z_center + max_range / 2)
        
        # Set equal aspect ratio
        self.ax.set_box_aspect([x_max - x_min, y_max - y_min, z_max - z_min])
        
        plt.suptitle('Best Fit Ellipsoid')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    PLOT = True
    USE_FOLDER=True
    # Load and animate the trajectory
    if not USE_FOLDER:
        FILES = [
            'simulations/20241126183708_sim_langevin_dt0.01_delta0.2120138165619025_km0.1_TC20_damping0.1.pkl'
        ]
    else:
        directory = "simulations"
        FILES = [os.path.join(directory, f) for f in os.listdir(directory)]

    extracted_data = {}
    for i, file in enumerate(FILES):
        cd = Path().resolve()
        filepath = cd / file

        sim_instance = MolecularDynamicsSimulation.load_state(filepath)
        last_state = sim_instance.state_trajectory[-1]
        
        positions = last_state['positions']
        positions = positions[:, 0, :]
        
        topology = last_state['topology']
        
        delta = sim_instance.delta
        num_monomers = topology.number_of_nodes()
        
        scaling = sim_instance.monomer_info['scaling']
        
        ellipsoid = Ellipsoid(positions, topology, scaling=scaling)
        ellipsoid.fit_ellipsoid_pca()
        surface_area_triangulation = ellipsoid.calc_surface_area(mode='triangulation')
        surface_area_fit = ellipsoid.calc_surface_area(mode='fit')
        
        axes_radii = np.array(ellipsoid.radii)
        
        #calculate axes ratio
        idx_axis_major = np.argmax(axes_radii)
        axis_major = axes_radii[idx_axis_major]
        
        mask = np.ones(len(axes_radii), dtype=bool)
        mask[idx_axis_major] = False
        axes_minor = axes_radii[mask] 
        
        #calculate mean and error from the two minor axes
        mean_axes_minor = np.mean(axes_minor)
        err_mean_minor = np.std(axes_minor, ddof=1)
        
        axes_ratio = axis_major/mean_axes_minor
        err_axes_ratio = axes_ratio*np.sqrt(err_mean_minor**2 / mean_axes_minor**2)
        
        extracted_data[i] = {'filename': filepath,
                             'delta': delta,
                             'num_monomers': num_monomers,
                             'axes_radii': axes_radii,
                             'axes_ratio': axes_ratio,
                             'err_axes_ratio': err_axes_ratio,
                             'surface_triangulation': surface_area_triangulation,
                             'surface_fit': surface_area_fit
                            }
       
        if PLOT:
            ellipsoid.plot()
       
    list_delta = [data['delta'] for data in extracted_data.values()]
    list_axes_ratios = [data['axes_ratio'] for data in extracted_data.values()]
    list_err_axes_ratios = [data['err_axes_ratio'] for data in extracted_data.values()]
    
    list_surface_triangulation = [data['surface_triangulation'] for data in extracted_data.values()]
    list_surface_fit = [data['surface_fit'] for data in extracted_data.values()]
        
    # Create a new figure for the 2D plot
    plt.figure()
    plt.errorbar(
        list_delta, list_axes_ratios,
        yerr=list_err_axes_ratios,
        fmt='o',            # Circle markers
        ecolor='grey',      # Error bar color
        capsize=5,          # Error bar cap size
        label='Data with Errors'
    )
    plt.xlabel('$\Delta$ (nm)')
    plt.ylabel('$a_{max}$/$a_{min}$')
    plt.title('Axes Ratio vs Delta')
    plt.show()
    
    # Create a new figure for the 2D plot
    plt.figure()
    plt.scatter(list_delta, list_surface_triangulation, label='Triangulation', marker='o')
    plt.scatter(list_delta, list_surface_fit, label='Fit', marker='x')
    plt.xlabel('$\Delta$ (nm)')
    plt.ylabel('Surface area (nm^2)')
    plt.legend()
    plt.title('Surface Area vs Delta')
    plt.show()