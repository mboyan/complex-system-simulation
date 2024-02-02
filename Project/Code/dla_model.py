"""
This module contains functions for a single simulation instance of a Diffusion-Limited Aggregation (DLA) model.
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt


# ===== Particle / lattice initialization =====

def init_seeds_bottom(lattice_size, n_seeds, n_dims=2):
    """
    Creates equally spaced seeds along a line at the bottom of the lattice.
    inputs:
        lattice_size (int) - size of the side of the lattice square 
        n_seeds (int) - the amount of seeds to generate
        n_dims (int) - the dimensions of the lattice; defaults to 2
    outputs:
        seed_coords (np.array) - an array of lattice site coordinates for the placement of initial seeds
    """
    assert 1 <= n_seeds <= lattice_size

    x_coords = np.delete(np.arange(0, n_seeds + 1), 0) * int(lattice_size/(n_seeds + 1))
    rest_coords = np.tile(np.zeros(n_seeds), (n_dims - 1, 1))
    # y_coords = np.zeros(n_seeds)

    # seed_coords = np.column_stack((x_coords, y_coords))
    seed_coords = np.insert(rest_coords, 0, x_coords.reshape((1, -1)), axis=0).T

    return seed_coords.astype(int)


def init_lattice(lattice_size, seed_coords):
    """
    Creates a square lattice with initial seeds placed on specific sites.
    inputs:
        lattice_size (int) - the size of the lattice along one dimension
        seed_coords (np.ndarray) - an array of lattice site coordinates for the placement of initial seeds
    """

    # Infer dimensions from number of seed coordinates
    lattice_dims = seed_coords.shape[1]
    assert lattice_dims > 1

    lattice = np.zeros(np.repeat(lattice_size, lattice_dims))
    
    seed_value = 1
    for seed_coord in seed_coords:
        lattice[tuple(seed_coord.T)] = seed_value
        seed_value += 1

    return lattice

def init_obstacle_lattice(lattice_size, boxes=None, seed_coords=None):
    """
    Create lattice where 0 determines free space and 1 determines an obstacle. 
    inputs:
        lattice_size (int) - the size of the lattice along one dimension
        boxes (np.ndarray) - a 2D array containing the diagonal corner points of rectangular obstacles in a flattened form (e.g. x1, x2, y1, y2, ...). Defaults to None
        seed_coords (np.ndarray) - the lattice coordinates of the initial seeds, used to check if seeds are in obstacles; defaults to None
    output:
        the obstacle lattice (np.ndarray) and the coordinates of the obstacle (np.ndarray)
    """

    # Infer dimensions from number of seed coordinates
    lattice_dims = seed_coords.shape[1]
    assert lattice_dims > 1

    if boxes is not None:
        assert np.ndim(boxes) == 2, 'box regions must be an np.ndarray with 2 dimensions'
        assert boxes.shape[1] == lattice_dims * 2, 'each box region must be a tuple of 2*lattice_dims points (e.g. x1, x2, y1, y2, ...)'

    obstacle_lattice = np.zeros(np.repeat(lattice_size, lattice_dims))

    # Make rectangle obstacles using four points
    if boxes is not None:
        # Create slices from the box coordinates and assign 1s to the obstacle lattice at these slices
        for i in range(0, boxes.shape[0]):
            slices = tuple([slice(boxes[i, j], boxes[i, j+lattice_dims]+1) for j in range(lattice_dims)])
            obstacle_lattice[slices] = 1

    # Check if there is a seed in the obstacle
    if np.any(obstacle_lattice[tuple(seed_coords.T)] == 1):
        print("At least one seed is inside an obstacle")

    # Maybe not necessary, might delete later
    # obstacle_locs = np.argwhere(obstacle_lattice == 1)

    return obstacle_lattice


def init_particles(lattice, prop_particles, obstacles=None):
    """
    Creates an array of n-dimensional particles where the number of particles
    is determined by a proportion from an input lattice size.
    inputs:
        lattice (np.ndarray) - an array of lattice sites containing 1's where there are seeds and 0's otherwise
        prop_particles (float) - a number between 0 and 1 determining the percentage / density of particles on the lattice
        obstacles (np.ndarray) - an array of lattice sites containing 1's where there are obstacles and 0's otherwise
    outputs:
        init_coords (np.ndarray) - an array of particle coordinates
    """
    assert 0 <= prop_particles <= 1, 'prop_articles must be a fraction of the particles'

    # Find empty locations in the lattice
    empty_locs = np.argwhere(lattice == 0)
    if type(obstacles) == np.ndarray:
        empty_locs = np.argwhere((lattice == 0) & (obstacles == 0))

    # Determine number of particles=
    n_particles = int(empty_locs.shape[0] * prop_particles)

    # if gravity:
    #     # Initialize particles in the top of the grid
    #     top = 2 * prop_particles
    #     init_coords = empty_locs[np.random.choice(int(top * empty_locs.shape[0]), size=n_particles, replace=False)]

    # else:
    #     # Initialize particles randomly wherever there are no seeds
    #     init_coords = empty_locs[np.random.choice(empty_locs.shape[0], size=n_particles, replace=False)]
    
    init_coords = empty_locs[np.random.choice(empty_locs.shape[0], size=n_particles, replace=False)]

    return init_coords


def regen_particles(lattice, n_particles, bndry_weights=None, obstacles=None):
    """
    Randomly regenerate a specific number of particles.
    inputs:
        lattice (np.ndarray) - an array of lattice sites containing 1's where there are seeds and 0's otherwise
        n_particles (int) - the number of particles to regenerate
        bndry_weights (np.ndarray) - an array of probabilities for regenerating particles at the boundaries in each dimension
        obstacles (np.ndarray) - an array of lattice sites containing 1's where there are obstacles and 0's otherwise; defaults to None
    """

    lattice_dims = np.ndim(lattice)
    assert len(set(lattice.shape)) == 1, 'lattice is not a square array'
    assert lattice_dims > 1

    # Find empty locations in the lattice
    empty_locs = np.argwhere(lattice == 0)

    # Only generates particles outside obstacles
    if type(obstacles) == np.ndarray:
        empty_locs = np.argwhere((lattice == 0) & (obstacles == 0))

    assert n_particles <= empty_locs.shape[0], 'too many particles to regenerate'

    # if gravity:
    #     # regenerate particles in the top of the grid
    #     regen_coords = empty_locs[np.random.choice(int(0.2 * empty_locs.shape[0]), size=n_particles, replace=False)]

    # else:
    #     # regenerate particles randomly wherever there are no seeds
    #     regen_coords = empty_locs[np.random.choice(empty_locs.shape[0], size=n_particles, replace=False)]
    
    if bndry_weights is not None:
        # Regenerate particles at the boundary
        slices = []

        # Create slices for selecting first and last row in each dimension
        for i in range(lattice_dims):
            start_slice = [slice(None)]*lattice_dims
            start_slice[i] = 0
            slices.append(tuple(start_slice))

            end_slice = [slice(None)]*lattice_dims
            end_slice[i] = -1
            slices.append(tuple(end_slice))
        
        # Pick a slice randomly based on the weights
        slice_selected = slices[np.random.choice(len(slices), p=bndry_weights.flatten())]

        # Generate indices for each dimension
        indices = [np.arange(size) for size in lattice.shape]

        # Combine the indices into a grid
        coords = np.stack(np.meshgrid(*indices, indexing='ij'), -1)

        # Pick random particle coordinates from the selected slice
        regen_coords = coords[slice_selected][np.random.choice(coords[slice_selected].shape[0], size=n_particles, replace=True)]

    else:
        # Regenerate particles randomly wherever there are no seeds
        regen_coords = empty_locs[np.random.choice(empty_locs.shape[0], size=n_particles, replace=False)]

    return regen_coords


def regen_probabilities(nbr_coords, nbr_weights):
    """
    Create boundary regeneration probabilities based on neighbourhood weights
    inputs:
        nbr_coords (np.ndarray) - an array of neighbourhood coordinates
        nbr_weights (np.ndarray) - an array of neighbourhood weights
    outputs:
        bndry_weights (np.ndarray) - an array of probabilities for regenerating particles at the boundaries in each dimension
    """
    assert nbr_coords.shape[0] == nbr_weights.shape[0], f'dimension mismatch between neighbourhood coordinates ({nbr_coords.shape[0]}) and weights ({nbr_weights.shape[0]})'
    
    # Infer dimensions from number of coordinates
    lattice_dims = nbr_coords.shape[1]
    
    dim_weights_start = [np.sum(nbr_weights[nbr_coords[:, dim] == 1]) for dim in range(lattice_dims)]
    dim_weights_end = [np.sum(nbr_weights[nbr_coords[:, dim] == -1]) for dim in range(lattice_dims)]
    bndry_weights = np.vstack((dim_weights_start, dim_weights_end)).T
    
    return bndry_weights


# ===== Particle movement functions =====

def move_particles_diffuse(particles_in, lattice, periodic=(False, True), moore=False, obstacles=None, drift_vec=None, regen_bndry=True):
    """
    Petrurbs the particles in init_array using a Random Walk.
    inputs:
        particles_in (numpy.ndarray) - an array of coordinates
        lattice (np.ndarray) - an array of lattice sites containing 1's where there are seeds and 0's otherwise
        periodic (tuple of bool) - defines whether the lattice is periodic in each dimension, number of elements must correspond to dimensions
        moore (bool) - determine whether the particles move in a Moore neighbourhood
            or otherwise in a von Neumann neighbourhood; defaults to False
        obstacles (np.ndarray) - an array of lattice sites containing 1's where there are obstacles and 0's otherwise; defaults to None
        drift_vec (np.ndarray) - a vector affecting the drift probabilities for each direction based on dot-product alignment; defaults to None
        regen_bndry (bool) - determines whether particles regenerate at the boundary or otherwise anywhere in space; defaults to True
    outputs:
        the particle array after one step (numpy.ndarray)
    """

    lattice_dims = np.ndim(lattice)
    lattice_size = lattice.shape[0]

    # Adjust periodicity tuple
    periodic = (periodic * (lattice_dims // len(periodic)) + periodic[:lattice_dims % len(periodic)]) if len(periodic) != 0 else ()

    assert lattice_dims > 1
    assert lattice_dims == particles_in.shape[1], f'dimension mismatch between lattice ({lattice_dims}) and particles ({particles_in.shape[1]})'
    # assert lattice_dims == len(periodic), 'dimension mismatch between lattice and periodicity tuple'
    assert len(set(lattice.shape)) == 1, 'lattice is not a square array'
    if obstacles is not None:
        assert lattice_dims == np.ndim(obstacles), f'dimension mismatch between lattice ({lattice_dims}) and obstacles ({np.ndim(obstacles)})'

    # Define Moore neighbourhood
    moves = [step for step in product([0, 1, -1], repeat=lattice_dims) if np.linalg.norm(step) != 0]
    moves = np.array(moves)

    # Reduce to von Neumann neighbourhood
    if not moore:
        moves = moves[np.sum(np.abs(moves), axis=1) == 1]
    # print('moves: ', moves)

    # Set boundary regeneration probabilities to None by default
    bndry_weights = None

    # Create perturbation vectors
    if drift_vec is not None:

        assert len(drift_vec) == lattice_dims, f'dimension mismatch between drift vector ({len(drift_vec)}) and lattice ({lattice_dims})'

        # Calculate weights for each attachment direction based on dot product with drift vector
        weights = np.dot(moves, drift_vec) + 1.0
        weights[weights < 0] = 0
        weights /= weights.sum()
        # print('weights drift: ', weights)

        perturbations = moves[np.random.choice(len(moves), particles_in.shape[0], p = weights)]

        # Create boundary regeneration probabilities based on drift probabilities
        if regen_bndry:
            bndry_weights = regen_probabilities(moves, weights)

    else:
        perturbations = moves[np.random.randint(len(moves), size=particles_in.shape[0])]

    # Move particles if on an unoccupied site
    mask = lattice[tuple(particles_in.T)] == 0
    particles_out = np.array(particles_in)
    particles_out[mask] += perturbations[mask]

    # Generate particle reserves
    if not np.all(np.array(periodic)):
        particles_regen = regen_particles(lattice, particles_in.shape[0], bndry_weights=bndry_weights, obstacles=obstacles)
    
    # Wrap around or regenerate
    for i, p in enumerate(periodic):
        if p:
            particles_out[:, i] = np.mod(particles_out[:, i], lattice_size)
        else:
            particles_out[:, i] = np.where(np.any((particles_out < 0) | (particles_out >= lattice_size), axis=1), particles_regen[:, i], particles_out[:, i])

    # For particles that have moved into an obstacle, revert them to their original positions.
    if type(obstacles) == np.ndarray:
        in_obstacles = obstacles[tuple(particles_out[mask].T)]
        particles_out[mask] = np.where(np.repeat(in_obstacles, lattice_dims).reshape(particles_out[mask].shape), particles_in[mask], particles_out[mask])
    
    return particles_out


# ===== Aggregation function =====

def aggregate_particles(particles, lattice, prop_particles=None, moore=False, obstacles=None, sun_vec=None, drift_vec=None, multi_seed=False):
    """
    Check if particles are neighbouring seeds on the lattice.
    If they are, place new seeds.
    inputs:
        particles (np.ndarray) - an array of particle coordinates
        lattice (np.ndarray) - an array of lattice sites containing 1's where there are seeds and 0's otherwise
        prop_particles (float) - a number between 0 and 1 determining the percentage / density of particles on the lattice;
            if None, the particles will not be regenerated to compensate the proportion
        moore (bool) - determine whether the neighbourhood is Moore or otherwise von Neumann; defaults to False
        obstacles (np.ndarray) - an array of lattice sites containing 1's where there are obstacles and 0's otherwise; defaults to None
        sun_vec (np.ndarray) - a vector affecting the growth direction by prioritising neighbours aligned with its direction;
            its magnitude affects how focused/diffuse the sunlight is: << 1 for diffuse, >> 1 for focused; defaults to [1, 0]
        drift_vec (np.ndarray) - a vector analogous to the drift vector in the move_particles_diffuse function;
            if supplied, a particle regeneration at the boundary is assumed; defaults to None
    """

    # Create a copy of the lattice
    lattice = np.array(lattice)

    lattice_dims = np.ndim(lattice)
    lattice_size = lattice.shape[0]
    assert lattice_dims > 1
    assert lattice_dims == particles.shape[1], f'dimension mismatch between lattice ({lattice_dims}) and particles ({particles.shape[1]})'
    assert len(set(lattice.shape)) == 1, 'lattice is not a square array'
    if sun_vec is not None:
        assert len(sun_vec) == lattice_dims, f'dimension mismatch between sun vector ({len(sun_vec)}) and lattice ({lattice_dims})'
    if obstacles is not None:
        assert lattice_dims == np.ndim(obstacles), f'dimension mismatch between lattice ({lattice_dims}) and obstacles ({np.ndim(obstacles)})'

    # Define particle neighbourhoods (Moore)
    nbrs = [neighbor for neighbor in product([0, 1, -1], repeat=lattice_dims) if np.linalg.norm(neighbor) != 0]
    nbrs = np.array(nbrs)

    # Reduce to von Neumann neighbourhood
    if not moore:
        nbrs = nbrs[np.sum(np.abs(nbrs), axis=1) == 1]
    # print('nbrs: ', nbrs)

    # Pad lattice with zeros (avoid periodic attachment)
    padded_lattice = np.pad(lattice, ((1, 1),)*lattice_dims, mode='constant')

    # Shift padded lattice by neighbours, then remove the padding
    shifted_lattices = np.array([np.roll(padded_lattice, shift, tuple(range(lattice_dims)))[(slice(1, -1),)*lattice_dims] for shift in nbrs])
    
    if multi_seed:

        nbr_vals = np.moveaxis(shifted_lattices, 0, -1)
        nbr_vals = np.where(nbr_vals == 0, 999999, nbr_vals)
        
        # Sort descending
        nbrs_sorted = np.sort(nbr_vals, axis=-1)
        indices_nbr = np.zeros(nbr_vals.shape, dtype=int)
        take_axis = np.take_along_axis(nbrs_sorted, indices_nbr, axis=-1)
        nbr_select = take_axis[...,0].astype(int)
        
        # reshape to get arrays of all possible neighbors for each point
        # reshaped_sl = shifted_lattices.reshape(shifted_lattices.shape[0], -1).astype(int)
    
        # # count most occurring neighbor
        # nbr_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength = int(shifted_lattices.max()+ 1)), axis = 0, arr = reshaped_sl)

        # # set count for zero to 0.5 (highest if it only zeros (no neighbors), lower when there is at least one neighbor)
        # nbr_counts[0] = 0.5

        # # find most occurring neighbor
        # most_occurring_nbrs = nbr_counts.argmax(axis = 0).reshape((lattice_size, lattice_size)) # picks the lowest value when equal amount of neighbors

    # Calculate weights for each attachment direction based on dot product with sun vector
    if sun_vec is not None:
        weights = np.dot(nbrs, -np.array(sun_vec)) + 1.0
        weights[weights < 0] = 0
    else:
        weights = np.ones(nbrs.shape[0])
    
    # Normalize weights
    weights /= np.sum(weights)
    # print('weights aggregation: ', weights)

    # Multiply shifted lattices by weights
    weights = np.repeat(weights, lattice_size ** lattice_dims)
    weights = np.reshape(weights, shifted_lattices.shape)
    occupied_flags = np.where(shifted_lattices > 0, 1.0, 0.0)
    occupied_flags *= weights
    summed_nbrs_lattice = np.sum(occupied_flags, axis=0)

    # Check if particles are neighbouring seeds
    u = np.random.uniform()
    new_seed_indices = np.argwhere(summed_nbrs_lattice[tuple(particles.T)] > np.max(weights) * u)

    # Update lattice
    if multi_seed:
        lattice[tuple(particles[new_seed_indices].T)] = nbr_select[tuple(particles[new_seed_indices].T)]
        # lattice[tuple(particles[new_seed_indices].T)] = most_occurring_nbrs[tuple(particles[new_seed_indices].T)]
        # lattice[tuple(particles[new_seed_indices].T)] = shifted_lattices[np.random]
    else:
        lattice[tuple(particles[new_seed_indices].T)] = 1

    # Compensate particle density
    if prop_particles is not None:
        # Recalculate lattice vacancy
        if obstacles is None:
            lattice_vacancy = np.argwhere(lattice == 0)
        else:
            lattice_vacancy = np.argwhere((lattice == 0) & (obstacles == 0))
        n_particles_potential = int(lattice_vacancy.shape[0] * prop_particles)

        # Regenerate as many particles as are needed to maintain the proportion
        mask = lattice[tuple(particles.T)] == 0
        active_particle_indices = np.flatnonzero(mask)
        inactive_particle_indices = np.flatnonzero(~mask)
        n_particles_deficit = n_particles_potential - active_particle_indices.shape[0]
        if n_particles_deficit > 0:

            if drift_vec is not None:
                # Create boundary regeneration probabilities based on drift probabilities
                bndry_weights = regen_probabilities(nbrs, weights)
            else:
                bndry_weights = None
            
            particles_regen = regen_particles(lattice, n_particles_deficit, bndry_weights=bndry_weights, obstacles=obstacles)
            particles[inactive_particle_indices[:n_particles_deficit]] = particles_regen

    return lattice, particles


# ===== Utility functions =====

def particles_to_lattice(particles, lattice_size):
    """
    Projects the particle coordinates on a lattice,
    returning a grid with 1's where there is a particle and 0's otherwise
    """
    assert np.max(particles) < lattice_size, 'mismatch between particle coordinates and lattice size'

    # Infer dimensions from number of particle coordinates
    lattice_dims = particles.shape[1]

    particle_lattice = np.zeros(np.repeat(lattice_size, lattice_dims))

    particle_lattice[tuple(particles.T)] = 1

    return particle_lattice