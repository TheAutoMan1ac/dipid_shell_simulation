import math
import numpy as np
import time
import traceback
import pickle
import copy
from pathlib import Path
from datetime import datetime
import random
import argparse
from numba import njit
import cProfile, pstats, io

import matplotlib.pyplot as plt

import networkx as nx
from collections import Counter
from scipy.spatial import KDTree

nx.config.cache_converted_graphs = False
# Set a fixed seed for Python's built-in random
#random.seed(78923486)
#np.random.seed(78923486)

np.seterr(divide='raise', over='raise', under='raise', invalid='raise')


# Assume k_B_T is given (e.g. k_B*T). If not needed, set k_B_T = 0.0.
k_B_T = 0.0  # set to a nonzero value for thermal noise


@njit
def update_accelerations_jit(positions, accelerations, mass, k1, k2, k3,
                             lengthEq, delta, a12, a23,
                             node_indices, neighbor_indices,
                             interlayer_distance):
    # Reset accelerations
    accelerations[:, :, :] = 0.0

    # Precompute inverse mass
    inv_mass = 1.0 / mass

    # Extract layer positions
    pos_top = positions[:, 0, :]
    pos_mid = positions[:, 1, :]
    pos_low = positions[:, 2, :]

    # Calculate self-layer forces
    forces_top_mid_self = calc_force_batch(pos_top, pos_mid, k2, interlayer_distance)
    forces_mid_low_self = calc_force_batch(pos_mid, pos_low, k2, interlayer_distance)

    # Update accelerations with self-layer forces
    accelerations[:, 0, :] += forces_top_mid_self * inv_mass
    accelerations[:, 1, :] += (-forces_top_mid_self + forces_mid_low_self) * inv_mass
    accelerations[:, 2, :] += -forces_mid_low_self * inv_mass

    # Extract node and neighbor layer positions
    pos_node_layers = positions[node_indices, :, :]
    pos_neigh_layers = positions[neighbor_indices, :, :]

    # Calculate inter-layer forces
    force_top = calc_force_batch(pos_node_layers[:, 0, :], pos_neigh_layers[:, 0, :], k1, lengthEq + 2.0 * delta)
    force_mid = calc_force_batch(pos_node_layers[:, 1, :], pos_neigh_layers[:, 1, :], k1, lengthEq)
    force_low = calc_force_batch(pos_node_layers[:, 2, :], pos_neigh_layers[:, 2, :], k1, lengthEq - 2.0 * delta)

    force_top_mid = calc_force_batch(pos_node_layers[:, 0, :], pos_neigh_layers[:, 1, :], k3, a12)
    force_mid_top = calc_force_batch(pos_node_layers[:, 1, :], pos_neigh_layers[:, 0, :], k3, a12)
    force_mid_low = calc_force_batch(pos_node_layers[:, 1, :], pos_neigh_layers[:, 2, :], k3, a23)
    force_low_mid = calc_force_batch(pos_node_layers[:, 2, :], pos_neigh_layers[:, 1, :], k3, a23)

    # Precompute forces divided by mass
    force_top *= inv_mass
    force_mid *= inv_mass
    force_low *= inv_mass
    force_top_mid *= inv_mass
    force_mid_top *= inv_mass
    force_mid_low *= inv_mass
    force_low_mid *= inv_mass

    N_pairs = node_indices.shape[0]

    for i in range(N_pairs):
        n_idx = node_indices[i]
        neigh_idx = neighbor_indices[i]

        # Cache references to accelerations to minimize array lookups
        acc_n0 = accelerations[n_idx, 0, :]
        acc_n1 = accelerations[n_idx, 1, :]
        acc_n2 = accelerations[n_idx, 2, :]
        acc_neigh0 = accelerations[neigh_idx, 0, :]
        acc_neigh1 = accelerations[neigh_idx, 1, :]
        acc_neigh2 = accelerations[neigh_idx, 2, :]

        # Same-layer forces
        acc_n0 += force_top[i]
        acc_neigh0 -= force_top[i]

        acc_n1 += force_mid[i]
        acc_neigh1 -= force_mid[i]

        acc_n2 += force_low[i]
        acc_neigh2 -= force_low[i]

        # Cross-layer forces
        acc_n0 += force_top_mid[i]
        acc_neigh1 -= force_top_mid[i]

        acc_n1 += force_mid_top[i]
        acc_neigh0 -= force_mid_top[i]

        acc_n1 += force_mid_low[i]
        acc_neigh2 -= force_mid_low[i]

        acc_n2 += force_low_mid[i]
        acc_neigh1 -= force_low_mid[i]

        # Assign the updated accelerations back
        accelerations[n_idx, 0, :] = acc_n0
        accelerations[n_idx, 1, :] = acc_n1
        accelerations[n_idx, 2, :] = acc_n2
        accelerations[neigh_idx, 0, :] = acc_neigh0
        accelerations[neigh_idx, 1, :] = acc_neigh1
        accelerations[neigh_idx, 2, :] = acc_neigh2


@njit
def calc_force_batch(v_r1_array, v_r2_array, k, a0):
    v_r12 = v_r1_array - v_r2_array  # Shape: (N, 3)
    l = np.sqrt((v_r12 * v_r12).sum(axis=1))  # Norm along axis=1

    mask_zero = (l == 0)
    l_safe = l.copy()
    l_safe[mask_zero] = 1.0

    v_r12_norm = v_r12 / l_safe[:, np.newaxis]
    v_r12_norm[mask_zero] = 0.0

    stretch = (l - a0)
    v_f = -k * stretch[:, np.newaxis] * v_r12_norm
    return v_f


# Helper function to generate random noise if needed
# If you do not want noise, set k_B_T = 0.0 or skip this step.
# @njit
# def generate_random_force(shape, gamma, mass, dt, k_B_T):
#     # shape: (N,3) for each layer, total shape: (N, L, 3)
#     # For simplicity, assume same distribution for all particles and layers:
#     sigma = 0.0
#     if k_B_T > 0.0:
#         # std deviation of random velocity increments
#         # Actually, random force: R(t). The velocity increment = R(t)*dt/mass
#         # R(t) should have std: sqrt(2 * gamma * k_B_T * m / dt)
#         sigma = np.sqrt(2.0 * gamma * mass * k_B_T / dt)

#     noise = np.zeros(shape, dtype=np.float64)
#     if sigma > 0.0:
#         for i in range(shape[0]):
#             for j in range(shape[1]):
#                 for k in range(3):
#                     noise[i, j, k] = np.random.randn() * sigma
#     return noise


@njit
def simulate_steps_jit(positions, positions_old, velocities, accelerations, 
                       mass, dt, damping_coeff, method_flag, 
                       k1, k2, k3, lengthEq, delta, a12, a23, interlayer_distance,
                       node_indices, neighbor_indices, n_inner_steps):
    # Convert damping_coeff to gamma (friction per unit mass or per unit time)
    # Assuming damping_coeff = gamma directly as a friction term with units 1/time.
    gamma = damping_coeff

    for _ in range(n_inner_steps):
        # Update accelerations
        update_accelerations_jit(
            positions,
            accelerations,
            mass,
            k1,
            k2,
            k3,
            lengthEq,
            delta,
            a12,
            a23,
            node_indices,
            neighbor_indices,
            interlayer_distance
        )

        if method_flag == 0:
            # Verlet (no damping, no noise)
            dt2 = dt * dt
            positions_new = 2.0 * positions - positions_old + accelerations * dt2
            positions_old[:] = positions
            positions[:] = positions_new

        else:
            # Langevin
            # Generate random force if needed (for temperature control)
            # If k_B_T = 0.0, this will just return zeros
            #R = generate_random_force(positions.shape, gamma, mass, dt, k_B_T)

            # Update velocities with friction and noise
            # v(t+dt) = v(t) + (a(t)-gamma v(t)) dt + R/m (with R scaled by dt if needed)
            # Here, R is force, so velocity increment from noise = (R / m) * dt
            # a(t)*dt = accelerations*dt
            # friction term = -gamma * velocities * dt
            # noise term = (R/m)*dt
            velocities[:] += accelerations * dt - gamma * velocities * dt #+ (R / mass) * dt

            # Update positions
            positions_old[:] = positions
            positions[:] = positions + velocities * dt


class MolecularDynamicsSimulation:
    # Initializations
    def __init__(self, dt, mass, lengthEq, delta, km, interlayer_distance=None, T_C=20, origin=np.zeros(3), method='langevin', damping_coeff=1, random_placement = False, random_chance = 0, monomer_info=None, batch_mode=False,alpha_deg=None,add_unit_every=None):
        #Filename generation
        #str_datetime = datetime.now().strftime("%Y%m%d%H%M%S") + f"{now.microsecond // 1000:03d}{now.microsecond % 1000:03d}"
        
        str_datetime = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = './simulations/' + str_datetime + '_sim_' + method + "_dt" + str(dt) + "_a" + str(alpha_deg) + "_delta" + str(delta) + "_km" + str(km)+'_TC' + str(T_C) + "_Bsize" + str(add_unit_every)
        if method == 'langevin':
            filename += "_damping" + str(damping_coeff)
        if random_placement:
            filename += "_random" + str(random_chance)
            
        filename += '.pkl'
        self.filename = filename

        #State memory
        self.state_trajectory = []
        self.sim_trajectory = []

        #Simulation parameters
        self.method = method
        if self.method == 'verlet':
            self.method_flag = 0
        else:
            self.method_flag = 1
            
        self.dt = np.float64(dt)
        self.T_C = T_C
        self.T_K = T_C + 273.15
        
        self.damping_coeff = damping_coeff
        
        #Particle characteristics
        if interlayer_distance is None:
            self.interlayer_distance = lengthEq
        else:
            self.interlayer_distance = interlayer_distance
        
        self.mass = np.float64(mass)
        
        self.lengthEq = np.float64(lengthEq)
        self.delta = np.float64(delta)
        
        self.k1 = np.float64(km)
        self.k2 = np.float64(km / 2)
        self.k3 = np.float64(km / 3)

        #Random placement characteristics (if activated)
        self.random_placement = random_placement
        self.random_chance = random_chance
        
        self.monomer_info = monomer_info
        self.batch_mode = batch_mode


        # Create initial topology
        self.topology = nx.Graph()
        self.top_boundaries = []
        
        # Initialize positions, velocities, accelerations, positions_old arrays
        # Shape: (N_nodes, 3_layers, 3)
        self.positions = np.zeros((0, 3, 3), dtype=np.float64)
        self.velocities = np.zeros((0, 3, 3), dtype=np.float64)
        self.accelerations = np.zeros((0, 3, 3), dtype=np.float64)
        self.positions_old = np.zeros((0, 3, 3), dtype=np.float64)
        
        # Map node IDs to indices
        self.map_node_config = {}
        self.node_counter = 0

        self.init_assembly_top_conf(interlayer_distance, origin)
        self.fix_velocity_distribution()
        
        # Empirically determine the equilibrium cross-layer distance
        # Distance between layer 1 and 2 -> a12
        # Distance between layer 2 and 3 -> a23
        self.a12 = np.linalg.norm(self.positions[0, 0, :] - self.positions[1, 1, :])
        self.a23 = np.linalg.norm(self.positions[0, 1, :] - self.positions[1, 2, :])
        
    def init_assembly_top_conf(self, interlayer_distance, origin):
        # Holds the initial 9 particles constituting the first unit cell (a triangular prism)
        positions = self.create_prims(interlayer_distance, origin)

        # Add initial particles
        for pos_layers in positions:
            self.add_particle(pos_layers)

        # Generate a complete topological graph from initial configuration
        topology_init = nx.complete_graph(self.topology.nodes())
        self.topology.add_edges_from(topology_init.edges())
        self.top_boundaries.append(set(self.topology.edges()))
        
    def init_vel_boltzmann(self, num_particles):
        sigma = np.sqrt(1.38e-23 * self.T_K / self.mass)
        vel = np.random.normal(0, sigma, (num_particles, 3)).astype(np.float64)
        return vel

    def fix_velocity_distribution(self):
        k_B = 1.38e-23
        N = self.topology.number_of_nodes() * 3  # Total number of particles (nodes * layers)

        v_total = np.sum(self.velocities)
        v_mean = v_total / N
        v_total -= v_mean

        # Step 4: Calculate the total squared velocity
        v_square_total = np.sum(self.velocities ** 2)
        v_square_mean = v_square_total / N

        # Step 6: Calculate the current kinetic energy and the desired kinetic energy
        kinetic_energy_per_particle = 0.5 * self.mass * v_square_mean
        desired_kinetic_energy = 1.5 * k_B * self.T_K
        scaling_factor = np.sqrt(desired_kinetic_energy / kinetic_energy_per_particle)

        self.velocities *= scaling_factor

    def create_prims(self, interlayer_distance, origin=np.zeros(3)):
        l1 = self.lengthEq + 2 * self.delta
        l2 = self.lengthEq
        l3 = self.lengthEq - 2 * self.delta

        ori_X = origin[0]
        ori_Y = origin[1]
        ori_Z = origin[2]

        h1 = l1 * math.sqrt(3) / 2
        h2 = l2 * math.sqrt(3) / 2
        h3 = l3 * math.sqrt(3) / 2

        u = np.linalg.norm(np.array([(l1 - l2)/2, (h1 - h2)/3], dtype=np.float64))
        dz = math.sqrt(interlayer_distance**2 - u**2)

        # Positions for each layer
        # Layer 0 (Top)
        pos_0 = np.array([
            [ori_X - l1 / 2, ori_Y - h1 / 3, ori_Z],
            [ori_X + l1 / 2, ori_Y - h1 / 3, ori_Z],
            [ori_X, ori_Y + h1 * 2 / 3, ori_Z]
        ], dtype=np.float64)

        # Layer 1 (Middle)
        pos_1 = np.array([
            [ori_X - l2 / 2, ori_Y - h2 / 3, ori_Z + dz],
            [ori_X + l2 / 2, ori_Y - h2 / 3, ori_Z + dz],
            [ori_X, ori_Y + h2 * 2 / 3, ori_Z + dz]
        ], dtype=np.float64)

        # Layer 2 (Bottom)
        pos_2 = np.array([
            [ori_X - l3 / 2, ori_Y - h3 / 3, ori_Z + 2 * dz],
            [ori_X + l3 / 2, ori_Y - h3 / 3, ori_Z + 2 * dz],
            [ori_X, ori_Y + h3 * 2 / 3, ori_Z + 2 * dz]
        ], dtype=np.float64)

        # Combine positions into a list of positions per node
        positions = []
        for i in range(3):  # Three nodes
            pos_layers = np.array([pos_0[i], pos_1[i], pos_2[i]], dtype=np.float64)
            positions.append(pos_layers)

        return positions


    # Particle handler
    def add_particle(self, positions):
        # Add a new particle with given positions for each layer
        self.topology.add_node(self.node_counter)
        
        # Expand arrays to accommodate the new node
        self.positions = np.vstack((self.positions, [positions]))
        self.velocities = np.vstack((self.velocities, [self.init_vel_boltzmann(3)]))
        self.accelerations = np.vstack((self.accelerations, np.zeros((1, 3, 3), dtype=np.float64)))
        self.positions_old = np.vstack((self.positions_old, [positions - self.velocities[-1] * self.dt]))
        
        new_node_id = self.node_counter
        self.map_node_config[new_node_id] = self.topology.number_of_nodes()-1
        self.node_counter += 1
        
        return new_node_id 

    def remove_particle(self, node):
        self.topology.remove_node(node)        
        self.map_node_config[node] = None
        
    def place_particle(self, edgeStart, edgeEnd):
        com_neighbours = list(nx.common_neighbors(self.topology, edgeStart, edgeEnd))
        if not com_neighbours:
            print("No common neighbor found.")
            return
        elif len(com_neighbours) >= 2:
            print(f"Trying to add a particle to an inner edge instead of a boundary edge! Common neighbours = {len(com_neighbours)}.")
        
        ref_node = com_neighbours[0]
        ref_node_idx = self.map_node_config[ref_node]
        node_1_idx = self.map_node_config[edgeStart]
        node_2_idx = self.map_node_config[edgeEnd]

        pos_ref = self.positions[ref_node_idx, :, :] 
        pos_node_1 = self.positions[node_1_idx, :, :]
        pos_node_2 = self.positions[node_2_idx, :, :]

        pos_new_layers = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            pos_ref_coords = pos_ref[i, :]
            pos_base1_coords = pos_node_1[i, :]
            pos_base2_coords = pos_node_2[i, :]

            coords = self.position_from_reference(pos_ref_coords, pos_base1_coords, pos_base2_coords)
            pos_new_layers[i, :] = coords

        new_node = self.add_particle(pos_new_layers)
        self.fix_velocity_distribution()

        self.topology.add_node(new_node)
        self.topology.add_edge(new_node, edgeStart)
        self.topology.add_edge(new_node, edgeEnd)
        
        return new_node
        
    def position_from_reference(self, p_ref, p_base1, p_base2):
        # Mirror point B across the line AC
        v_base = p_base2 - p_base1
        v_ref = p_ref - p_base1
        v_base_norm = v_base / np.linalg.norm(v_base)
        proj_length = np.dot(v_ref, v_base_norm)
        proj_point = p_base1 + proj_length * v_base_norm
        
        new_point = 2 * proj_point - p_ref
        
        return new_point

    def getParticleCount(self):
        return self.topology.number_of_nodes()

    def getNodeStatistics(self):
        if self.topology.is_directed():
            raise TypeError("The function expects an undirected graph. Please provide an undirected NetworkX graph.")
    
        # Extract degrees of all nodes
        degrees = [degree for _, degree in self.topology.degree()]        
        degree_counts = Counter(degrees)
        degree_distribution = dict(sorted(degree_counts.items()))
        
        return degree_distribution
    

    def calc_angle(self, node_id, neighbour1_id, neighbour2_id):
        pos_node = self.positions[self.map_node_config[node_id], 0, :]
        pos_neighbour1 = self.positions[self.map_node_config[neighbour1_id], 0, :]
        pos_neighbour2 = self.positions[self.map_node_config[neighbour2_id], 0, :]

        v1 = pos_neighbour1 - pos_node
        v2 = pos_neighbour2 - pos_node

        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = dot_product / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)
        angle_degrees = np.degrees(angle)

        return angle_degrees
    #################################### Debugging #############################
    # def calc_force_batch(self, v_r1_array, v_r2_array, k, a0):
    #     v_r12 = v_r1_array - v_r2_array  # Shape: (N, 3)
    #     l = np.linalg.norm(v_r12, axis=1)  # Shape: (N,)

    #     # Prevent division by zero
    #     l_safe = np.where(l == 0, 1, l)

    #     v_r12_norm = v_r12 / l_safe[:, np.newaxis]  # Shape: (N, 3)
    #     # Adjust force where l == 0 to zero
    #     v_r12_norm[l == 0] = 0

    #     v_f = -k * (l - a0)[:, np.newaxis] * v_r12_norm  # Shape: (N, 3)
    #     return v_f
    
    # def calcTotalEnergy(self):
    #     totalEnergy = 0

    #     # Build interaction lists for neighbor interactions
    #     node_indices = []
    #     neighbor_indices = []

    #     for edge in self.topology.edges():
    #         node_id = edge[0]
    #         neighbour_id = edge[1]
    #         node_indices.append(self.map_node_config[node_id])
    #         neighbor_indices.append(self.map_node_config[neighbour_id])

    #     node_indices = np.array(node_indices)
    #     neighbor_indices = np.array(neighbor_indices)

    #     # Positions of interacting pairs
    #     pos_node_layers = self.positions[node_indices, :, :]  # Shape: (N_pairs, 3, 3)
    #     pos_neigh_layers = self.positions[neighbor_indices, :, :]

    #     # Same-layer potential energy (k1 interactions)
    #     energy_top = 0.5 * self.k1 * (np.linalg.norm(pos_node_layers[:, 0, :] - pos_neigh_layers[:, 0, :], axis=1) -
    #                                 (self.lengthEq + 2 * self.delta)) ** 2
    #     energy_mid = 0.5 * self.k1 * (np.linalg.norm(pos_node_layers[:, 1, :] - pos_neigh_layers[:, 1, :], axis=1) -
    #                                 self.lengthEq) ** 2
    #     energy_low = 0.5 * self.k1 * (np.linalg.norm(pos_node_layers[:, 2, :] - pos_neigh_layers[:, 2, :], axis=1) -
    #                                 (self.lengthEq - 2 * self.delta)) ** 2

    #     # Cross-layer potential energy (k3 interactions)
    #     energy_top_mid = 0.5 * self.k3 * (np.linalg.norm(pos_node_layers[:, 0, :] - pos_neigh_layers[:, 1, :], axis=1) -
    #                                     self.a12) ** 2
    #     energy_mid_top = 0.5 * self.k3 * (np.linalg.norm(pos_node_layers[:, 1, :] - pos_neigh_layers[:, 0, :], axis=1) -
    #                                     self.a12) ** 2
    #     energy_mid_low = 0.5 * self.k3 * (np.linalg.norm(pos_node_layers[:, 1, :] - pos_neigh_layers[:, 2, :], axis=1) -
    #                                     self.a23) ** 2
    #     energy_low_mid = 0.5 * self.k3 * (np.linalg.norm(pos_node_layers[:, 2, :] - pos_neigh_layers[:, 1, :], axis=1) -
    #                                     self.a23) ** 2

    #     # Sum same-layer and cross-layer energies
    #     energy_interactions = energy_top + energy_mid + energy_low + energy_top_mid + energy_mid_top + energy_mid_low + energy_low_mid
    #     totalEnergy += np.sum(energy_interactions)

    #     # Self-interactions (forces between layers within the same node)
    #     pos_top = self.positions[:, 0, :]  # Shape: (N_nodes, 3)
    #     pos_mid = self.positions[:, 1, :]
    #     pos_low = self.positions[:, 2, :]

    #     energy_top_mid_self = 0.5 * self.k2 * (np.linalg.norm(pos_top - pos_mid, axis=1) - self.interlayer_distance) ** 2
    #     energy_mid_low_self = 0.5 * self.k2 * (np.linalg.norm(pos_mid - pos_low, axis=1) - self.interlayer_distance) ** 2

    #     # Sum self-interaction energies
    #     energy_self = energy_top_mid_self + energy_mid_low_self
    #     totalEnergy += np.sum(energy_self)

    #     # Debugging: Print all energy components
    #     print("----- Energy Breakdown -----")
    #     print(f"Energy Top (k1): {np.sum(energy_top):.4f}")
    #     print(f"Energy Mid (k1): {np.sum(energy_mid):.4f}")
    #     print(f"Energy Low (k1): {np.sum(energy_low):.4f}")
    #     print(f"Energy Top-Mid (k3): {np.sum(energy_top_mid):.4f}")
    #     print(f"Energy Mid-Top (k3): {np.sum(energy_mid_top):.4f}")
    #     print(f"Energy Mid-Low (k3): {np.sum(energy_mid_low):.4f}")
    #     print(f"Energy Low-Mid (k3): {np.sum(energy_low_mid):.4f}")
    #     print(f"Energy Self Top-Mid (k2): {np.sum(energy_top_mid_self):.4f}")
    #     print(f"Energy Self Mid-Low (k2): {np.sum(energy_mid_low_self):.4f}")
    #     print(f"Total Energy: {totalEnergy:.4f}")
    #     print("-----------------------------\n")

    #     return totalEnergy

    
    def calcTotalEnergy(self):
        totalEnergy = 0

        # Build interaction lists for neighbor interactions
        node_indices = []
        neighbor_indices = []

        for edge in self.topology.edges():
            node_id = edge[0]
            neighbour_id = edge[1]
            node_indices.append(self.map_node_config[node_id])
            neighbor_indices.append(self.map_node_config[neighbour_id])

        node_indices = np.array(node_indices)
        neighbor_indices = np.array(neighbor_indices)

        # Positions of interacting pairs
        pos_node_layers = self.positions[node_indices, :, :]  # Shape: (N_pairs, 3, 3)
        pos_neigh_layers = self.positions[neighbor_indices, :, :]

        # Same-layer potential energy (k1 interactions)
        energy_top = 0.5 * self.k1 * (np.linalg.norm(pos_node_layers[:, 0, :] - pos_neigh_layers[:, 0, :], axis=1) -
                                      (self.lengthEq + 2 * self.delta)) ** 2
        energy_mid = 0.5 * self.k1 * (np.linalg.norm(pos_node_layers[:, 1, :] - pos_neigh_layers[:, 1, :], axis=1) -
                                      self.lengthEq) ** 2
        energy_low = 0.5 * self.k1 * (np.linalg.norm(pos_node_layers[:, 2, :] - pos_neigh_layers[:, 2, :], axis=1) -
                                      (self.lengthEq - 2 * self.delta)) ** 2

        # Cross-layer potential energy (k3 interactions)
        energy_top_mid = 0.5 * self.k3 * (np.linalg.norm(pos_node_layers[:, 0, :] - pos_neigh_layers[:, 1, :], axis=1) -
                                          self.a12) ** 2
        energy_mid_top = 0.5 * self.k3 * (np.linalg.norm(pos_node_layers[:, 1, :] - pos_neigh_layers[:, 0, :], axis=1) -
                                          self.a12) ** 2
        energy_mid_low = 0.5 * self.k3 * (np.linalg.norm(pos_node_layers[:, 1, :] - pos_neigh_layers[:, 2, :], axis=1) -
                                          self.a23) ** 2
        energy_low_mid = 0.5 * self.k3 * (np.linalg.norm(pos_node_layers[:, 2, :] - pos_neigh_layers[:, 1, :], axis=1) -
                                          self.a23) ** 2

        totalEnergy += np.sum(energy_top + energy_mid + energy_low +
                              energy_top_mid + energy_mid_top + energy_mid_low + energy_low_mid)

        # Self-interactions (forces between layers within the same node)
        pos_top = self.positions[:, 0, :]  # Shape: (N_nodes, 3)
        pos_mid = self.positions[:, 1, :]
        pos_low = self.positions[:, 2, :]

        energy_top_mid_self = 0.5 * self.k2 * (np.linalg.norm(pos_top - pos_mid, axis=1) - self.interlayer_distance) ** 2
        energy_mid_low_self = 0.5 * self.k2 * (np.linalg.norm(pos_mid - pos_low, axis=1) - self.interlayer_distance) ** 2

        totalEnergy += np.sum(energy_top_mid_self + energy_mid_low_self)

        return totalEnergy


    # Check if there is still a boundary in the assembly -> if not, the surface must be closed
    def is_closed_surface(self):
        return sum(len(s) for s in self.top_boundaries) <= 3 and self.getParticleCount() > 3         
        
    @staticmethod
    def update_accelerations(sim_instance):
        edges = sim_instance.topology.edges()
        node_indices = []
        neighbor_indices = []
        for edge in edges:
            node_indices.append(sim_instance.map_node_config[edge[0]])
            neighbor_indices.append(sim_instance.map_node_config[edge[1]])

        node_indices = np.array(node_indices, dtype=np.int64)
        neighbor_indices = np.array(neighbor_indices, dtype=np.int64)

        update_accelerations_jit(
            sim_instance.positions,
            sim_instance.accelerations,
            sim_instance.mass,
            sim_instance.k1,
            sim_instance.k2,
            sim_instance.k3,
            sim_instance.lengthEq,
            sim_instance.delta,
            sim_instance.a12,
            sim_instance.a23,
            node_indices,
            neighbor_indices,
            sim_instance.interlayer_distance
        )

    def close_pentamer(self, node_id, neighbour_1, neighbour_2):
        # Remove the two edges connected to neighbour_2 from boundary_edges
        edge1 = tuple(sorted((node_id, neighbour_1)))
        edge2 = tuple(sorted((node_id, neighbour_2)))
        
        current_boundary = None
        for distinct_boundary in self.top_boundaries:
            if edge1 in distinct_boundary and edge2 in distinct_boundary:
                current_boundary = distinct_boundary
        
        current_boundary.discard(edge1)
        current_boundary.discard(edge2)

        # Remove boundary edges involving neighbour_2
        edges_to_remove = {edge for edge in current_boundary if neighbour_2 in edge}
        current_boundary -= edges_to_remove

        new_boundaries = {tuple(sorted((node, neighbour_1))) for edge in edges_to_remove for node in edge if node != neighbour_2}
        current_boundary.update(new_boundaries)

        # Add new edges and update boundary_edges
        for neighbour_id in list(self.topology.neighbors(neighbour_2)):
            self.topology.add_edge(neighbour_1, neighbour_id)
                
        # Remove node and update data structures
        self.remove_particle(neighbour_2)

    def close_hexamer(self, min_entry):
        # Close as Hexamer
        edge1 = tuple(sorted((min_entry['node_id'], min_entry['neighbour_1'])))
        edge2 = tuple(sorted((min_entry['node_id'], min_entry['neighbour_2'])))
        
        current_boundary = None
        for distinct_boundary in self.top_boundaries:
            if edge1 in distinct_boundary and edge2 in distinct_boundary:
                current_boundary = distinct_boundary
        
        current_boundary.discard(edge1)
        current_boundary.discard(edge2)

        # Add new edge and update boundary_edges
        self.topology.add_edge(min_entry['neighbour_1'], min_entry['neighbour_2'])
        new_edge = tuple(sorted((min_entry['neighbour_1'], min_entry['neighbour_2'])))
        if new_edge in current_boundary:
            current_boundary.discard(new_edge)
        else:
            current_boundary.add(new_edge)

    def next_position(self):
        # Use boundary edges to find nodes at the boundary
        boundary_nodes = set()
        for distinct_boundary in self.top_boundaries:
            for edge in distinct_boundary:
                boundary_nodes.add(edge[0])
                boundary_nodes.add(edge[1])
       
        # Iterate through all nodes at the boundary and calculate the angle between edges
        growing_edge_positions = []
        for node in boundary_nodes:
            degree = self.topology.degree(node)
            for distinct_boundary in self.top_boundaries:
                connected_edges = [edge for edge in distinct_boundary if node in edge]
                connected_ids = [edge[1] if edge[0] == node else edge[0] for edge in connected_edges]
                
                if len(connected_ids) < 2:
                    continue  # Skip nodes that don't have at least two boundary connections
                
                angle = self.calc_angle(node, connected_ids[0], connected_ids[1])
                if degree == 2 or degree == 3:
                    angle = 360.00 - angle

                growing_edge_positions.append({'angle': round(angle,2), 'node_id': node, 'degree': degree, 'neighbour_1': connected_ids[0], 'neighbour_2': connected_ids[1]})

        if not growing_edge_positions:
            print("No valid positions to add particles.")
            return

        # If randomness is activated, check if random placement should happen
        best_candidate = None
        if self.random_placement and random.random() < self.random_chance:
            best_candidate = random.choice(list(growing_edge_positions))

        # Find next position by getting smallest closing angle (if ambiguous choose randomly) <-> equal to get energetically best position on growing edge
        else:
            min_angle = min(growing_edge_positions, key=lambda x: x['angle'])
            min_entries = [x for x in growing_edge_positions if x['angle'] == min_angle['angle']]

            if min_entries and len(min_entries) > 1:
                best_candidate = random.choice(min_entries)
            else:
                best_candidate = min_entries[0]

        if best_candidate['angle'] < 30:
            self.close_pentamer(best_candidate['node_id'], best_candidate['neighbour_1'], best_candidate['neighbour_2'])

        elif 30 <= best_candidate['angle'] <= 60 and best_candidate['degree'] == 6:
            self.close_hexamer(best_candidate)
            
        elif best_candidate['degree'] == 7:   
            self.close_pentamer(best_candidate['node_id'], best_candidate['neighbour_1'], best_candidate['neighbour_2'])
            
        else:
            new_node = self.place_particle(best_candidate['node_id'], best_candidate['neighbour_1'])

            edge = tuple(sorted((best_candidate['node_id'], best_candidate['neighbour_1'])))
            
            current_boundary = None
            for distinct_boundary in self.top_boundaries:
                if edge in distinct_boundary:
                    current_boundary = distinct_boundary

            current_boundary.discard(edge)
            current_boundary.add(tuple(sorted((new_node, best_candidate['node_id']))))
            current_boundary.add(tuple(sorted((new_node, best_candidate['neighbour_1']))))


    def cleanup_map_node_config(self):
        idx_none = [idx for idx, (_, value) in enumerate(self.map_node_config.items()) if value is None]
        
        self.positions = np.delete(self.positions, idx_none, axis=0)
        self.velocities = np.delete(self.velocities, idx_none, axis=0)
        self.accelerations = np.delete(self.accelerations, idx_none, axis=0)
        self.positions_old = np.delete(self.positions_old, idx_none, axis=0)
        
        map_new = {}
        counter = 0
        for node_id, idx in self.map_node_config.items():
            if idx is not None:
                map_new[node_id] = counter
                counter += 1
                
        self.map_node_config = map_new

    def check_closure_event(self, min_topo_dist=5, max_physical_dist=0.8):
        closure_occurred = False

        self.cleanup_map_node_config()

        # Prepare positions and KDTree for spatial queries
        positions = self.positions[:, 0, :]  # Top layer positions
        kd_tree = KDTree(positions)
        idx_to_node_id = [node_id for node_id, _ in self.map_node_config.items()]

        # Identify valid boundary nodes
        boundary_nodes = set()
        for distinct_boundary in self.top_boundaries:
            nodes = {node for edge in distinct_boundary for node in edge}
            boundary_nodes.update(nodes)

        for node in boundary_nodes:
            idx = self.map_node_config[node]
            pos = positions[idx]

            # Find the closest other node within max_physical_dist
            distances, indices = kd_tree.query(pos, k=2)
            dist, idx = distances[1], indices[1]  # Skip the node itself (distances[0])

            if dist <= max_physical_dist:
                other_node = idx_to_node_id[idx]
                if other_node in self.map_node_config and nx.has_path(self.topology, node, other_node):
                    topo_dist = nx.shortest_path_length(self.topology, source=node, target=other_node)
                    if topo_dist >= min_topo_dist:
                        if not self.batch_mode:
                            print('FIXING EVENT OCCURRED')

                        current_boundary = None
                        for neighbor in list(self.topology.neighbors(other_node)):
                            self.topology.add_edge(node, neighbor)
                            old_edge = tuple(sorted((other_node, neighbor)))
                            for distinct_boundary in self.top_boundaries:
                                if old_edge in distinct_boundary:
                                    current_boundary = distinct_boundary
                                    current_boundary.discard(old_edge)
                                    
                                    new_edge = tuple(sorted((node, neighbor)))
                                    current_boundary.add(new_edge)

                        # Remove other_node from data structures
                        current_boundary = {edge for edge in current_boundary if other_node not in edge}
                        self.remove_particle(other_node)
                        
                        self.top_boundaries = self.decompose_boundary(node)
                        closure_occurred = True
                        return closure_occurred

    def decompose_boundary(self, start_node):
        # Normalize edges to sorted tuples for consistency
        remaining_edges = self.top_boundaries[0].copy()
        cycles = []
        
        while True:
            # Find an edge that includes the start_node
            start_edge = next((edge for edge in remaining_edges if start_node in edge), None)
            if not start_edge:
                break  # No more cycles can be formed starting from start_node

            # Initialize the new cycle
            cycle = set()
            cycle.add(start_edge)
            remaining_edges.remove(start_edge)

            # Determine the next node to visit
            current_node = start_edge[1] if start_edge[0] == start_node else start_edge[0]
            prev_node = start_node

            while current_node != start_node:
                # Find the next edge connected to current_node, excluding the edge we came from
                next_edge = next(
                    (edge for edge in remaining_edges if current_node in edge and (edge[0] != prev_node and edge[1] != prev_node)),
                    None
                )

                if not next_edge:
                    # Dead end reached; incomplete cycle
                    break

                # Add the edge to the current cycle
                cycle.add(next_edge)
                remaining_edges.remove(next_edge)

                # Update nodes for the next iteration
                prev_node, current_node = current_node, next_edge[1] if next_edge[0] == current_node else next_edge[0]

            # Check if a complete cycle was formed
            if current_node == start_node:
                cycles.append(cycle)
            else:
                # If not a complete cycle, you might want to handle it differently
                # For simplicity, we'll ignore incomplete cycles here
                pass

        return cycles

    def check_degree_overload(self, max_degree):
        for _, degree in self.topology.degree():
            if degree >= max_degree:
                return True
        
        return False

    def remove_minimal_cycles(self):
        if self.getParticleCount() == 3:
            return
        
        for distinct_boundary in self.top_boundaries:
            G = nx.Graph()
            G.add_edges_from(distinct_boundary)
            cycles = nx.cycle_basis(G)
            
            # Filter for triangular cycles (cycles with exactly 3 nodes)
            triangular_cycles = [cycle for cycle in cycles if len(cycle) == 3]

            # Convert each triangular cycle into edges and remove from original edges
            for cycle in triangular_cycles:
                # Convert the cycle nodes to edges
                cycle_edges = {tuple(sorted((cycle[i], cycle[(i+1) % 3]))) for i in range(3)}
                distinct_boundary -= cycle_edges
            
        return
    
    

    # Load and save
    def save_state_simulation(self):
        cd = Path(__file__).parent.resolve()

        # Prepare the new state
        state = {
            'positions': copy.deepcopy(self.positions),
            'topology': copy.deepcopy(self.topology),
            'positions_old': copy.deepcopy(self.positions_old),
            'velocities': copy.deepcopy(self.velocities),
            'accelerations': copy.deepcopy(self.accelerations),
            'boundaries': copy.deepcopy(self.top_boundaries),
            'map_node_config': copy.deepcopy(self.map_node_config),
        }
        self.state_trajectory.append(state)

        # Only pickle self
        with open(cd / self.filename, 'wb') as f:
            pickle.dump(self, f)

    def save_state_trajectory(self):
        cd = Path(__file__).parent.resolve()

        # Prepare the new state
        state = {
            'positions': copy.deepcopy(self.positions),
            'topology': copy.deepcopy(self.topology),
            'positions_old': copy.deepcopy(self.positions_old),
            'velocities': copy.deepcopy(self.velocities),
            'accelerations': copy.deepcopy(self.accelerations),
            'boundaries': copy.deepcopy(self.top_boundaries),
            'map_node_config': copy.deepcopy(self.map_node_config),
        }
        self.state_trajectory.append(state)

        filename_trajectory = ''.join(str(self.filename).split('.')[0:-1]) + '_trajectory.pkl'

        # Only pickle self
        with open(cd / filename_trajectory, 'wb') as f:
            pickle.dump(self.state_trajectory, f)

    @classmethod
    def load_state(cls, filename, filename_append='', start_at=-1):
        with open(filename, 'rb') as f:
            loaded_instance = pickle.load(f)

        trajectory = copy.deepcopy(loaded_instance.state_trajectory[start_at])

        # Update the filename
        loaded_instance.filename = '.'.join(loaded_instance.filename.split('.')[:-1]) + '_' + filename_append + '.pkl'

        # Update the state
        loaded_instance.positions = trajectory['positions']
        loaded_instance.positions_old = trajectory['positions_old']
        loaded_instance.velocities = trajectory['velocities']
        loaded_instance.accelerations = trajectory['accelerations']
        loaded_instance.topology = trajectory['topology']
        loaded_instance.top_boundaries = trajectory['boundaries']
        loaded_instance.map_node_config = trajectory['map_node_config']

        return loaded_instance

    # Simulation
    def verlet_update(self):
        # Recalculate forces and update accelerations
        self.update_accelerations(self)

        # Update positions using Verlet algorithm
        dt2 = self.dt ** 2
        positions_new = 2 * self.positions - self.positions_old + self.accelerations * dt2
        self.positions_old[:] = self.positions.copy()
        self.positions[:] = positions_new

    def langevin_update(self):
        # Recalculate forces and update accelerations
        self.update_accelerations(self)

        self.positions_old[:] = self.positions.copy()
        self.positions += (self.accelerations * self.mass * self.dt) / self.damping_coeff

    def simulate_step(self):
        if self.method == 'verlet':
            self.verlet_update()
        elif self.method == 'langevin':
            self.langevin_update()
        else:
            raise Exception("No valid method chosen")

class SimulationVisualizer:
    def __init__(self, sim, scaling=1, plot_outer_layer=False):
        self.sim = sim
        self.scaling = scaling
        self.plot_outer_layer = plot_outer_layer
        self.stop_simulation = False  # Flag to indicate if the simulation should stop

        # Initialize the plot
        self.initialize_plot()
        
    def initialize_plot(self):
        plt.close('all')
        
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1, 1, 1])

        # Connect the close event handler
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        # Initial particle positions
        if self.plot_outer_layer:
            vectors = self.sim.positions[:, 0, :]
        else:
            vectors = self.sim.positions.reshape(-1, 3)
        self.xs = vectors[:, 0]*self.scaling
        self.ys = vectors[:, 1]*self.scaling
        self.zs = vectors[:, 2]*self.scaling

        # Scatter plot for particles
        self.scatter = self.ax.scatter(self.xs, self.ys, self.zs, color='blue', s=20, label='Particles')

        # Store references to the edge lines
        self.edge_lines = []

        for edge in self.sim.topology.edges():
            plot_layers = 1 if self.plot_outer_layer else 3

            for j in range(plot_layers):
                pos1 = self.sim.positions[self.sim.map_node_config[edge[0]], j, :]*self.scaling
                pos2 = self.sim.positions[self.sim.map_node_config[edge[1]], j, :]*self.scaling

                edge_tuple = tuple(sorted(edge))
                edge_color_palette = ['red', 'magenta', 'orange']
                color = 'black'
                for i, distinct_boundary in enumerate(self.sim.top_boundaries):
                    if edge_tuple in distinct_boundary:
                        color = edge_color_palette[i]

                line, = self.ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], color=color)
                self.edge_lines.append(line)

        # Set labels and title
        if self.scaling == 1:
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('y')
            self.ax.set_zlabel('z')
        else:
            self.ax.set_xlabel('X (nm)')
            self.ax.set_ylabel('Y (nm)')
            self.ax.set_zlabel('Z (nm)')
            
        self.ax.set_title('Capsid Assembly Simulation')

    def on_close(self, event):
        print('Figure closed')
        # Set a flag to stop the simulation
        self.stop_simulation = True

    def update_plot(self):
        # Update particle positions
        if self.plot_outer_layer:
            vectors = self.sim.positions[:, 0, :]
        else:
            vectors = self.sim.positions.reshape(-1, 3)
        self.xs = vectors[:, 0]*self.scaling
        self.ys = vectors[:, 1]*self.scaling
        self.zs = vectors[:, 2]*self.scaling

        # Update the scatter plot for particles
        self.scatter._offsets3d = (self.xs, self.ys, self.zs)

        # Remove old edge lines
        while self.edge_lines:
            line = self.edge_lines.pop()
            line.remove()

        for edge in self.sim.topology.edges():
            plot_layers = 1 if self.plot_outer_layer else 3

            for j in range(plot_layers):
                pos1 = self.sim.positions[self.sim.map_node_config[edge[0]], j, :]*self.scaling
                pos2 = self.sim.positions[self.sim.map_node_config[edge[1]], j, :]*self.scaling

                edge_tuple = tuple(sorted(edge))
                edge_color_palette = ['red', 'magenta', 'orange']
                color = 'black'
                for i, distinct_boundary in enumerate(self.sim.top_boundaries):
                    if edge_tuple in distinct_boundary:
                        color = edge_color_palette[i]

                # Draw the new edge and store the line reference
                line, = self.ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], color=color)
                self.edge_lines.append(line)

        # Dynamically adjust the axis limits based on particle positions
        margin = 0  # Extra margin to ensure the particles are fully visible
        x_min, x_max = min(self.xs) - margin, max(self.xs) + margin
        y_min, y_max = min(self.ys) - margin, max(self.ys) + margin
        z_min, z_max = min(self.zs) - margin, max(self.zs) + margin

        # Ensure the x, y, and z axes have the same range (same extensions)
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        max_range = max(x_range, y_range, z_range)

        # Center the axes by adjusting the minimum and maximum to keep them equal in range
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        x_min_new = x_center - max_range / 2
        x_max_new = x_center + max_range / 2
        y_min_new = y_center - max_range / 2
        y_max_new = y_center + max_range / 2
        z_min_new = z_center - max_range / 2
        z_max_new = z_center + max_range / 2

        # Set the new limits with consistent x, y, and z ranges
        self.ax.set_xlim(x_min_new, x_max_new)
        self.ax.set_ylim(y_min_new, y_max_new)
        self.ax.set_zlim(z_min_new, z_max_new)

        # Redraw the plot to show updates
        self.fig.canvas.draw()
        plt.pause(0.1)

# Main simulation loop

def run_simulation(sim, visualizer, n_steps, add_unit_every, save_every_batch, plot_every_batch, save_what, max_degree=12,equilibrium_threshold=1e-6, equilibrium_threshold_absolute=1e-12):
    start_time = time.time()
    batch_size = add_unit_every
    n_batches = n_steps // batch_size
    remainder = n_steps % batch_size
    #n_rounds = 0

    # Parameters for equilibration
    equil_window_size = batch_size  # for example, 2000 steps

    for batch_idx in range(n_batches):
        # Save state if needed
        if batch_idx != 0 and batch_idx % save_every_batch == 0:
            if save_what == 'simulation':
                sim.save_state_simulation()
            elif save_what == 'trajectory':
                sim.save_state_trajectory()

        # Check if surface is closed
        if sim.is_closed_surface():
            # Equilibrate if needed
            # Note: If the surface is closed, we assume no further addition of nodes or main steps.
            # Just finalize and return.
            if visualizer is not None:
                visualizer.update_plot()
            if save_what == 'simulation':
                sim.save_state_simulation()
            elif save_what == 'trajectory':
                sim.save_state_trajectory()

            total_time = time.time() - start_time
            hours, remainder_time = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder_time, 60)
            print('SIMULATION FINISHED')
            steps_done = batch_idx * batch_size
            print(f'{steps_done} steps were simulated in {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds.')
            num_particles = sim.getParticleCount()
            print(f'{num_particles} nodes were added.')
            degree_distribution = sim.getNodeStatistics()
            for degree, count in degree_distribution.items():
                print(f'NODES OF DEGREE {degree} = {count}')
            if visualizer is not None:
                plt.ioff()
                plt.show()
            return

        # Check closure event
        closure_occurred = sim.check_closure_event(min_topo_dist=5, max_physical_dist=0.5)

        # If no closure, add a new node after the first batch
        if batch_idx != 0 and not closure_occurred:
            sim.next_position()

        # If a node was just added or a closure event occurred, equilibrate the system
        # Build node/neighbor arrays once before equilibrium loop
        node_indices = []
        neighbor_indices = []
        for edge in sim.topology.edges():
            node_indices.append(sim.map_node_config[edge[0]])
            neighbor_indices.append(sim.map_node_config[edge[1]])
        node_indices = np.array(node_indices, dtype=np.int64)
        neighbor_indices = np.array(neighbor_indices, dtype=np.int64)

        # Perform equilibrium steps until the system is stable
        # Equilibration is needed if we just added a node or if a closure occurred
        equilibrated = False

        while not equilibrated:
            #energy_old = sim.calcTotalEnergy()
            energy_per_node_old = sim.calcTotalEnergy() / sim.getParticleCount()
            # Run a batch of equilibrium steps
            simulate_steps_jit(sim.positions,
                            sim.positions_old,
                            sim.velocities,
                            sim.accelerations,
                            sim.mass,
                            sim.dt,
                            sim.damping_coeff,
                            sim.method_flag,
                            sim.k1,
                            sim.k2,
                            sim.k3,
                            sim.lengthEq,
                            sim.delta,
                            sim.a12,
                            sim.a23,
                            sim.interlayer_distance,
                            node_indices,
                            neighbor_indices,
                            equil_window_size)

            # Compute energy per node
            energy_per_node_new = sim.calcTotalEnergy() / sim.getParticleCount()

            drift = abs(energy_per_node_new - energy_per_node_old)
            if drift <= equilibrium_threshold * abs(energy_per_node_old) or drift < equilibrium_threshold_absolute:
                equilibrated = True
                #print(f"pass after n_rounds: {n_rounds}\n with energy: {energy_per_node_new}, drif: {drift}")
                #n_rounds = 0
            else:
                #n_rounds += 1
                energy_per_node_old = energy_per_node_new
                #print(f"NOT pass after n_rounds: {n_rounds}\n with energy: {energy_per_node_new}, drif: {drift}")

        # Post-equilibration checks
        sim.remove_minimal_cycles()

        if batch_idx % 10 == 0:
            if sim.check_degree_overload(max_degree):
                #print(f'Node degree overload encountered (DEGREE >= {max_degree}). Aborting simulation.')
                degree_distribution = sim.getNodeStatistics()
                for degree, count in degree_distribution.items():
                    print(f'NODES OF DEGREE {degree} = {count}')
                if visualizer is not None:
                    plt.ioff()
                    plt.show()
                return

        if batch_idx % plot_every_batch == 0 and visualizer is not None:
            visualizer.update_plot()

    # Handle remainder steps if needed
    # According to the instructions, no main batch integration is needed here.
    # Just do simple steps if the remainder is nonzero.
    if remainder > 0:
        for i in range(remainder):
            sim.simulate_step()

    # Final save
    if save_what == 'simulation':
        sim.save_state_simulation()
    elif save_what == 'trajectory':
        sim.save_state_trajectory()
    return


def get_sim_params_from_dipid(r, h, alpha_sticky_deg, printout=True):
    angle_sticky_rad = np.radians(alpha_sticky_deg)
    
    l_T = h*np.cos(np.pi/2 - angle_sticky_rad) + 3.94
    
    a_0 = 2*(r*np.cos(angle_sticky_rad) + l_T)
    r_container = a_0/(2*np.sin(angle_sticky_rad))
    a_2 = 2*(r_container - h)*np.sin(angle_sticky_rad)

    a_eq = (a_0 + a_2)/2
    delta_eq = (a_0 - a_2)/4
    interlayer_distance = h/2
    
    scaling = a_eq
    
    a_eq_sim = a_eq/scaling
    delta_eq_sim = delta_eq/scaling
    interlayer_distance_sim = interlayer_distance/scaling
    
    if printout:
        print(f'Used parameters are:')
        print(f'Radius r={r}nm')
        print(f'Height h={h}nm')
        print(f'Half binding angle: alpha_sticky_deg={alpha_sticky_deg}°')
        print()
        print(f'Simulation equilibrium length: a_eq_sim={a_eq_sim}')
        print(f'Simulation equilibrium asymmetry: delta_eq_sim={delta_eq_sim}')
        print(f'Interlayer distance: interlayer_distance_sim={interlayer_distance_sim}')
        print(f'Scaling factor sim_units -> physical_units: scaling={scaling}')
    
    return a_eq_sim, delta_eq_sim, interlayer_distance_sim, scaling

if __name__ == '__main__':
#def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run molecular dynamics simulation with specified parameters.")
    parser.add_argument('--alpha_sticky_deg', type=float, default=15, help="Alpha sticky degree (default: 15)")
    parser.add_argument('--save_every_batch', type=int, default=1, help="Steps interval to save simulation state (default: 250)")
    parser.add_argument('--plot_every_batch', type=int, default=1, help="Steps interval to plot simulation state (default: 250)")
    parser.add_argument('--n_steps', type=int, default=10000000, help="Total number of simulation steps (default: 10000000)")
    parser.add_argument('--batch_mode', action='store_true', help="Run simulation in batch mode without plotting")
    parser.add_argument('--random_placement', action='store_true', help="Place monomers randomly with random_chance")
    parser.add_argument('--random_chance', type=float, default=0.005, help="Chance of randomly placing a monomer")
    parser.add_argument('--add_unit_every', type=int, default=40000, help="Chance of randomly placing a monomer")


    args = parser.parse_args()

    # Simulation parameters
    # FIXED PARAMETERS
    MASS = 1
    T_C = 20
    PLOT_OUTER_LAYER = True
    DT = 0.05
    METHOD = 'langevin'
    KM = 0.1 
    DAMPING_COEFFICIENT = np.sqrt(4*MASS*KM)
    EQUILIBRIUM_THRESHOLD = 1e-6
    EQUILIBRIUM_THRESHOLD_ABSOLUTE = 1e-15
    
    # VARIABLE SIMULATION PARAMETERS
    random_placement = args.random_placement
    random_chance = args.random_chance

    # DIPID PARAMETERS
    r = 14.25 #nm
    h = 18 #nm
    alpha_deg = args.alpha_sticky_deg
    #l_sticky = np.tan(np.radians(alpha_sticky_deg))* h_dipid #nm 

    # RUN FLAVOUR
    batch_mode = args.batch_mode

    # DYNAMIC PARAMETERS
    A0, DELTA, INTERLAYER_DISTANCE, SCALING = get_sim_params_from_dipid(r, h, alpha_deg, True)
    
    # Packing DIPID info and passing to Simulation class to use later in analysis
    MONOMER_INFO = {
        'radius': r,
        'height': h,
        'alpha_binding': alpha_deg,
        'scaling': SCALING
    }

    sim = MolecularDynamicsSimulation(
        dt=DT,
        mass=MASS,
        lengthEq=A0,
        delta=DELTA,
        km=KM,
        interlayer_distance=INTERLAYER_DISTANCE,
        T_C=T_C,
        method=METHOD,
        damping_coeff=DAMPING_COEFFICIENT,
        random_placement=random_placement,
        random_chance=random_chance,
        monomer_info=MONOMER_INFO,
        batch_mode=batch_mode,
        alpha_deg=alpha_deg,
        add_unit_every=args.add_unit_every
        
    )
    
    if not args.batch_mode:
        visualizer = SimulationVisualizer(sim, scaling=SCALING, plot_outer_layer=PLOT_OUTER_LAYER)
    else:
        visualizer = None  # No visualizer in batch mode

    n_steps = args.n_steps
    add_unit_every = args.add_unit_every
    save_every_batch = args.save_every_batch
    plot_every_batch = args.plot_every_batch

    run_simulation(sim, visualizer, n_steps, add_unit_every, save_every_batch, plot_every_batch, 'simulation',equilibrium_threshold=EQUILIBRIUM_THRESHOLD, equilibrium_threshold_absolute = EQUILIBRIUM_THRESHOLD_ABSOLUTE)
    # try:
    #     run_simulation(sim, visualizer, n_steps, add_unit_every, save_every_batch, plot_every_batch, 'simulation',equilibrium_threshold=EQUILIBRIUM_THRESHOLD, equilibrium_threshold_absolute = EQUILIBRIUM_THRESHOLD_ABSOLUTE)
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     traceback.print_exc()
    pass
    
    
    # Load and continue simulation if needed
    '''
    try:
        n_steps = 1000000
        add_unit_every = 250
        save_every = args.save_every
        plot_every = args.plot_every
        
        filename = './Simulation/simulations/test_case_1.pkl'
        newsim = MolecularDynamicsSimulation.load_state(filename, 'cont1', start_at=-40)
        visualizer = SimulationVisualizer(newsim, plot_outer_layer=PLOT_OUTER_LAYER)
        run_simulation(newsim, visualizer, n_steps, add_unit_every, save_every, plot_every, 'simulation')
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    '''
    

# if __name__ == '__main__':
#     pr = cProfile.Profile()
#     pr.enable()

#     main()  #

#     pr.disable()
#     s = io.StringIO()
#     sortby = 'cumulative'  # Options: 'time', 'cumulative', etc.
#     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#     ps.print_stats(20)  # Print top 20 functions

#     print(s.getvalue())
