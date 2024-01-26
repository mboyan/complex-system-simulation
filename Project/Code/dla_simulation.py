"""
This module contains functions for setting up and running series of DLA simulations.
"""

import numpy as np
from itertools import product

import dla_model as dm
import cs_measures as cm

def run_dla(lattice_size, max_timesteps, seeds, particle_density, target_mass=None, **sim_params):
    """
    Run a single DLA simulation.
    inputs:
        lattice_size (int) - the size of the lattice along one dimension
        time_steps (int) - the number of time steps for the simulation
        max_timesteps (int) - the maximum number of time steps for the simulation
        seeds (np.ndarray) - an array of lattice site coordinates for the placement of initial seeds
        particle_density (float) - a number between 0 and 1 determining the percentage / density of particles on the lattice
        target_mass (int) - if specified, the number of DLA cells to reach before stopping the simulation. Defaults to None
        track_radius (bool) - if True, track the maximum radius of the DLA structure from a single initial seed over time. Defaults to False
        sim_params (dict) - a dictionary of simulation parameters
    outputs:
        lattice_frames (np.ndarray) - a series of lattice states
        particles_frames (np.ndarray) - a series of particle coordinates
        dla_radii (np.ndarray) - a series of maximum DLA radii
    """
    

    # Unpack simulation parameters
    periodic = sim_params.get('periodic', (False, True))
    move_moore = sim_params.get('move_moore', False)
    aggr_moore = sim_params.get('aggr_moore', False)
    regen_toggle = sim_params.get('regen_toggle', False)
    track_radius = sim_params.get('track_radius', False)

    assert (track_radius == True and seeds.shape[0] == 1) or track_radius == False, 'radius tracking only works for single seed simulations'
    
    # Enable regeneration of particles
    if regen_toggle:
        regen_density = particle_density
    else:
        regen_density = None

    # Infer dimensions from number of seed coordinates
    lattice_dims = seeds.shape[1]

    # Initialize lattice
    lattice_start = dm.init_lattice(lattice_size, seeds)

    # Initialize particles
    particle_density = 0.1
    particles_start = dm.init_particles(lattice_start, particle_density)

    # Arrays for storing data frames
    lattice_frames = np.empty([max_timesteps] + [lattice_size for _ in range(lattice_dims)])
    particles_frames = np.empty_like(lattice_frames)
    dla_radii = np.empty(max_timesteps)

    current_lattice = np.array(lattice_start)
    current_particles = np.array(particles_start)
    max_radius = 2
    dla_radii[0] = max_radius

    # Growth loop
    for step in range(max_timesteps):
        
        # Record current state
        lattice_frames[step] = np.array(current_lattice)
        particles_frames[step] = dm.particles_to_lattice(current_particles, lattice_size)

        # Move particles
        current_particles = dm.move_particles_diffuse(current_particles, current_lattice, periodic=periodic, moore=move_moore)

        # Aggregate particles
        new_lattice, current_particles = dm.aggregate_particles(current_particles, current_lattice, regen_density, periodic=periodic, moore=aggr_moore)
        
        # Calculate maximum radius of DLA structure
        if track_radius and step > 0:
            # Check difference between new state and old state
            new_aggregate_indices = np.argwhere(new_lattice - current_lattice)
            if new_aggregate_indices.shape[0] > 0:
                seed_compare = np.repeat(seeds[0], new_aggregate_indices.shape[0])
                seed_compare = seed_compare.reshape(new_aggregate_indices.shape)
                new_radius = np.max(np.linalg.norm(new_aggregate_indices - seed_compare, axis=1))
                if new_radius > max_radius:
                    max_radius = new_radius
            dla_radii[step] = max_radius

        current_lattice = np.array(new_lattice)

        # Check for stopping conditions
        if target_mass is not None and np.sum(current_lattice) >= target_mass:
            break

    # Check if target mass was reached
    assert np.sum(current_lattice) >= target_mass, 'target mass not reached within given time steps'

    # Trim frames
    lattice_frames = lattice_frames[:step + 1]
    particles_frames = particles_frames[:step + 1]
    dla_radii = dla_radii[:step + 1]

    return lattice_frames, particles_frames, dla_radii


def analyse_fractal_dimension(n_sims, lattice_size_series, max_timesteps_series, seeds_series, particle_density_series, target_mass_series, radius_scale_mode=False, n_saved_sims=1, **sim_params):
    """
    Runs a series of simulations to estimate the mean fractal dimension of DLA structures.
    inputs:
        n_sims (int) - the number of simulations to run for each parameter combination
        lattice_size_series (np.ndarray) - a series of lattice sizes
        max_timesteps_series (np.ndarray) - a series of maximum time steps
        seeds_series (np.ndarray) - a series of seed coordinates
        particle_density_series (np.ndarray) - a series of particle densities
        target_mass_series (np.ndarray) - a series of target masses
        n_saved_sims (int) - the number of DLA evolutions to save for each parameter combination. Defaults to 1
        radius_scale_mode (bool) - if True, use the maximum radius of the DLA structure as a scale reference for the fractal dimension. Defaults to False
        sim_params (dict) - a dictionary of simulation parameters
    outputs:
        sim_results (pd.DataFrame) - a dataframe of simulation results
        dla_evolutions (dict) - a dictionary of saved DLA evolutions in the form of n-D lattice series
            containing the aggregate states and particle positions over time
    """

    assert lattice_size_series.ndim == 1, 'lattice_size_series must be a 1D array'
    assert max_timesteps_series.ndim == 1, 'max_timesteps_series must be a 1D array'
    assert seeds_series.ndim == 3, 'seeds_series must be a 3D array'
    assert particle_density_series.ndim == 1, 'particle_density_series must be a 1D array'
    assert target_mass_series.ndim == 1, 'target_mass_series must be a 1D array'
    assert np.all(np.log2(lattice_size_series) % 1 == 0), 'lattice_size_series must be a series of powers of 2'
    assert n_saved_sims <= n_sims, 'n_saved_sims must be less than or equal to n_sims'

    # Initialize dataframe for storing simulation results
    sim_results = []
    dla_evolutions = {'lattice_frames': [], 'particles_frames': []}
    
    param_combos = product(lattice_size_series, max_timesteps_series, seeds_series, particle_density_series, target_mass_series)

    # Iterate over parameter combinations
    for i, combo in enumerate(param_combos):

        # Unpack and print simulation parameters
        param_names = ['lattice_size', 'max_timesteps', 'seeds', 'particle_density', 'target_mass']
        lattice_size, max_timesteps, seeds, particle_density, target_mass = combo
        param_string = "; ".join([f"{k}: {v}" for k, v in zip(param_names, combo)])
        print(f'Running parameters: {param_string}')

        # Run simulations
        for j in range(n_sims):

            # Extract radius track toggle from sim_params
            track_radius = sim_params.get('track_radius', False)
            assert (track_radius and radius_scale_mode) or (not radius_scale_mode), 'radius tracking must be enabled for radius scale mode'

            print(f'Running simulation {j + 1} of {n_sims}')

            # Run DLA simulation
            lattice_frames, particles_frames, dla_radii = run_dla(lattice_size, max_timesteps, seeds, particle_density, target_mass, **sim_params)

            # Save simulation steps
            evol_ref = None
            if j < n_saved_sims:
                dla_evolutions['lattice_frames'].append(lattice_frames)
                dla_evolutions['particles_frames'].append(particles_frames)
                evol_ref = i * n_sims + j

            # Compute fractal dimension of current simulation
            if radius_scale_mode:
                scale_series = dla_radii
                n_box_series = np.sum(lattice_frames, axis=tuple(range(1, lattice_frames.ndim)))
                dim_box_series, coeffs = cm.fractal_dimension_radius(dla_radii, n_box_series)
            else:
                dim_box_series, scale_series, n_box_series, coeffs = cm.fractal_dimension(lattice_frames[-1], lattice_size)

            # Save simulation results
            new_data = {'lattice_size': lattice_size, 'max_timesteps': max_timesteps, 'seeds': list(seeds), 'particle_density': particle_density, 'target_mass': target_mass,
                             'dim_box_series': dim_box_series, 'scale_series': scale_series, 'n_box_series': n_box_series, 'coeffs': coeffs, 'evol_ref': evol_ref}
            sim_results.append(new_data)
    
    return sim_results, dla_evolutions