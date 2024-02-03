"""
This module contains functions for setting up and running series of DLA simulations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import math
import scipy.stats as stat

import dla_model as dm
import cs_measures as csm
import vis_tools as vt
import powerlaw as pl

def run_dla(lattice_size, max_timesteps, particle_density, n_seeds=int(1), target_mass=None, drift_vec=np.array([0,-1]), sun_vec=np.array([0,-1]), obstacle_boxes=None, **sim_params):
    """
    Run a single DLA simulation.
    inputs:
        lattice_size (int) - the size of the lattice along one dimension
        time_steps (int) - the number of time steps for the simulation
        max_timesteps (int) - the maximum number of time steps for the simulation
        particle_density (float) - a number between 0 and 1 determining the percentage / density of particles on the lattice
        n_seeds (int) - number of seeds to generate at the bottom of the lattice
        target_mass (int) - if specified, the number of DLA cells to reach before stopping the simulation; defaults to None
        obstacle_boxes (np.ndarray) - an array of obstacle box coordinates (diagonal corners in flattened form, e.g. x1, x2, y1, y2, ...); defaults to None
        drift_vec (np.ndarray) - a vector determining the drift direction of particles (movement direction bias); defaults to [1,0]
        sun_vec (np.ndarray) - a vector determining the direction of the sun (growth direction bias); defaults to [1,0]
        sim_params (dict) - a dictionary of simulation parameters which are constant among all simulations
    outputs:
        lattice_frames (np.ndarray) - a series of lattice states
        particles_frames (np.ndarray) - a series of particle coordinates
        dla_radii (np.ndarray) - a series of maximum DLA radii
        seeds (np.ndarray) - the array of initial seeds
    """
    
    assert type(n_seeds) == int, 'number of seeds must be an integer'
    assert n_seeds > 0, 'number of seeds must be positive'

    # Unpack fixed simulation parameters
    periodic = sim_params.get('periodic', (False, True))
    move_moore = sim_params.get('move_moore', False)
    aggr_moore = sim_params.get('aggr_moore', False)
    regen_mode = sim_params.get('regen_mode', None)
    track_radius = sim_params.get('track_radius', False)

    assert (track_radius == True and n_seeds == 1) or track_radius == False, 'radius tracking only works for single seed simulations'
    assert (regen_mode in [None, 'edge', 'anywhere']), 'regen_mode must be None, "edge" or "anywhere"'
    assert np.ndim(drift_vec) == np.ndim(sun_vec), 'sun and drift vectors must have the same number of dimensions'
    
    # Enable regeneration of particles
    regen_bndry = False
    if regen_mode is not None:

        regen_density = particle_density
        
        if regen_mode == 'edge':
            regen_bndry = True
    else:
        regen_density = None

    # Infer dimensions from number of drift vector coordinates
    lattice_dims = drift_vec.shape[0]

    # Initialize a predefined number of seeds at the bottom of the lattice
    seeds = dm.init_seeds_bottom(lattice_size, n_seeds, lattice_dims)
    # print(f'Initial seeds: {seeds}')

    # Initialize lattice
    lattice_start = dm.init_lattice(lattice_size, seeds)

    # Initialize obstacle lattice
    if obstacle_boxes is not None:
        obstacles = dm.init_obstacle_lattice(lattice_size, boxes=obstacle_boxes, seed_coords=seeds)
    else:
        obstacles = None

    # Initialize particles
    particle_density = 0.1
    particles_start = dm.init_particles(lattice_start, particle_density, obstacles)

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

        # print(f'Running time step {step + 1} of {max_timesteps}')
        
        # Record current state
        lattice_frames[step] = np.array(current_lattice)
        particles_frames[step] = dm.particles_to_lattice(current_particles, lattice_size)

        # Move particles
        current_particles = dm.move_particles_diffuse(current_particles, current_lattice, periodic=periodic, moore=move_moore, obstacles=obstacles,
                                                      drift_vec=drift_vec, regen_bndry=regen_bndry)

        # Aggregate particles
        new_lattice, current_particles = dm.aggregate_particles(current_particles, current_lattice, regen_density, moore=aggr_moore,
                                                                obstacles=obstacles, sun_vec=sun_vec)
        
        # Calculate maximum radius of DLA structure
        if track_radius and step > 0:
            # Check difference between new state and old state
            new_aggregate_indices = np.argwhere(new_lattice - current_lattice)
            if new_aggregate_indices.shape[0] > 0:
                seed_compare = np.tile(seeds[0], new_aggregate_indices.shape[0])
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
    if target_mass is not None:
        if np.sum(current_lattice) < target_mass:
            print(f'Target mass not reached by {target_mass - np.sum(current_lattice)} within given time steps!')

    # Trim frames
    lattice_frames = lattice_frames[:step + 1]
    particles_frames = particles_frames[:step + 1]
    dla_radii = dla_radii[:step + 1]

    return lattice_frames, particles_frames, dla_radii, seeds


def analyse_dla_patterns(n_sims, lattice_size_series, max_timesteps_series, n_seeds_series, particle_density_series,
                         target_mass_series, drift_vec_series, sun_vec_series, obstacle_box_series,
                         fdim_measure=None, calc_branch_distr=False, calc_mass=False, n_saved_sims=1, **sim_params):
    """
    Runs a series of simulations to estimate the mean fractal dimension of DLA structures.
    inputs:
        n_sims (int) - the number of simulations to run for each parameter combination
        lattice_size_series (np.ndarray) - a series of lattice sizes
        max_timesteps_series (np.ndarray) - a series of maximum time steps
        particle_density_series (np.ndarray) - a series of particle densities
        n_seeds_series (np.ndarray) - a series of seed numbers
        target_mass_series (np.ndarray) - a series of target masses
        fdim_measure (str) - selects the fractal dimension measuring method, can be None / 'cgrain' (coarse-graining) / 'radius' (radius-based scale) / 'both'
        calc_branch_distr (bool) - if True, compute the branch distribution of the DLA structure; defaults to False
        calc_mass (bool) - if True, compute the mass of the DLA structure over time; defaults to False
        n_saved_sims (int) - the number of DLA evolutions to save for each parameter combination; defaults to 1
        sim_params (dict) - a dictionary of simulation parameters
    outputs:
        sim_results (pd.DataFrame) - a dataframe of simulation results
        dla_evolutions (dict) - a dictionary of saved DLA evolutions in the form of n-D lattice series
            containing the aggregate states and particle positions over time
    """

    assert lattice_size_series.ndim == 1, 'lattice_size_series must be a 1D array'
    assert max_timesteps_series.ndim == 1, 'max_timesteps_series must be a 1D array'
    assert n_seeds_series.ndim == 1, 'seeds_series must be a 1D array'
    assert particle_density_series.ndim == 1, 'particle_density_series must be a 1D array'
    assert target_mass_series.ndim == 1, 'target_mass_series must be a 1D array'
    assert obstacle_box_series.ndim == 1 or obstacle_box_series.ndim == 3, 'obstacle_box_series must be a 1D or 3D array'
    assert drift_vec_series.ndim == 2, 'drift_vec_series must be a 2D array'
    assert sun_vec_series.ndim == 2, 'sun_vec_series must be a 2D array'
    assert np.all(np.log2(lattice_size_series) % 1 == 0), 'lattice_size_series must be a series of powers of 2'
    assert n_saved_sims <= n_sims, 'n_saved_sims must be less than or equal to n_sims'
    assert fdim_measure in [None, 'cgrain', 'radius', 'both'], 'fdim_measure must be None, "cgrain", "radius" or "both"'
    assert (calc_branch_distr and np.any(n_seeds_series) == 1) or calc_branch_distr == False, 'branch distribution only works for single seed simulations'
    assert (fdim_measure in ['radius', 'both'] and calc_mass) or fdim_measure not in ['radius', 'both'], 'mass calculation must be enabled for radius scale mode'
    assert (fdim_measure in ['radius', 'both'] and np.any(n_seeds_series) == 1) or fdim_measure not in ['radius', 'both'], 'radius scale mode only works for single seed simulations'

    # Enable radius tracking if fdim_measure is 'radius' or 'both'
    if fdim_measure == 'radius' or fdim_measure == 'both':
        sim_params['track_radius'] = True

    # Initialize dataframe for storing simulation results
    sim_results = []
    dla_evolutions = {'lattice_frames': [], 'particles_frames': []}
    
    param_combos = product(lattice_size_series, max_timesteps_series, particle_density_series, n_seeds_series, target_mass_series,
                           obstacle_box_series, drift_vec_series, sun_vec_series)

    param_combos_list = list(param_combos)
    print(f"Number of possible parameter combinations: {len(list(param_combos_list))}")

    # Iterate over parameter combinations
    for i, combo in enumerate(param_combos_list):

        # Unpack and print simulation parameters
        param_names = ['lattice_size', 'max_timesteps', 'particle_density', 'n_seeds', 'target_mass', 'obstacle_boxes', 'drift_vec', 'sun_vec']
        lattice_size, max_timesteps, particle_density, n_seeds, target_mass, obstacle_boxes, drift_vec, sun_vec = combo
        param_string = "; ".join([f"{k}: {v}" for k, v in zip(param_names, combo)])
        print(f'Running parameters: {param_string}')

        # Run simulations
        for j in range(n_sims):

            print(f'Running simulation {j + 1} of {n_sims}')

            # Run DLA simulation
            lattice_frames, particles_frames, dla_radii, seeds = run_dla(lattice_size, max_timesteps, particle_density, n_seeds, target_mass,
                                                                  drift_vec, sun_vec, obstacle_boxes, **sim_params)

            # Save simulation steps
            evol_ref = None
            if j < n_saved_sims:
                dla_evolutions['lattice_frames'].append(lattice_frames)
                dla_evolutions['particles_frames'].append(particles_frames)
                evol_ref = i * n_saved_sims + j

            # Initialize measures
            sim_measures = {'mass_series': None, 'fdr_scale_series': None, 'fdr_n_box_series': None, 'fdr_dim_box_series': None, 'fdr_coeffs': None,
                            'fdc_dim_box_series': None, 'fdc_scale_series': None, 'fdc_n_box_series': None, 'fdc_coeffs': None,
                            'branch_lengths_unique': None, 'branch_length_counts': None, 'branches': None}

            # Compute mass over time
            if calc_mass:
                mass_series = np.sum(lattice_frames, axis=tuple(range(1, lattice_frames.ndim)))
                sim_measures['mass_series'] = mass_series

            # Compute fractal dimension of current simulation
            if fdim_measure == 'radius' or fdim_measure == 'both':
                sim_measures['fdr_scale_series'] = dla_radii
                sim_measures['fdr_n_box_series'] = np.array(mass_series)
                fdr = csm.fractal_dimension_radius(dla_radii, mass_series)
                sim_measures['fdr_dim_box_series'] = fdr[0]
                sim_measures['fdr_coeffs'] = fdr[1]
            if fdim_measure == 'cgrain' or fdim_measure == 'both':
                fdc = csm.fractal_dimension_clusters(lattice_frames[-1])
                sim_measures['fdc_dim_box_series'] = fdc[0]
                sim_measures['fdc_scale_series'] = fdc[1]
                sim_measures['fdc_n_box_series'] = fdc[2]
                sim_measures['fdc_coeffs'] = fdc[3]
            
            # Compute branch distribution of current simulation
            if calc_branch_distr:
                branch_distribution = csm.branch_distribution(lattice_frames[-1], seeds[0], moore=sim_params['aggr_moore'])
                sim_measures['branch_lengths_unique'] = branch_distribution[0]
                sim_measures['branch_length_counts'] = branch_distribution[1]
                sim_measures['branches'] = branch_distribution[2]
            
            # Save simulation results
            new_data = {'lattice_size': lattice_size, 'max_timesteps': max_timesteps, 'seeds': list(seeds),
                        'particle_density': particle_density, 'target_mass': target_mass,
                        'drift_vec': drift_vec, 'sun_vec': sun_vec, 'obstacle_boxes': obstacle_boxes,
                        'sim_measures': sim_measures, 'evol_ref': evol_ref}
            sim_results.append(new_data)
    
    sim_results = pd.DataFrame(sim_results)

    return sim_results, dla_evolutions


def convert_to_tuple(nested_list):
    """
    Converts a nested list to a tuple.
    inputs:
        nested_list (list) - a list that may contain other lists
    outputs:
        nested_tuple (tuple) - a tuple with the same elements as nested_list, but with all lists converted to tuples
    """
    if nested_list is not None:
        if isinstance(nested_list, (float, int, str)):
            return nested_list
        else:
            return tuple(convert_to_tuple(i) if (isinstance(i, np.ndarray) or isinstance(i, list)) else i for i in nested_list)
    else:
        return None


def analyse_sim_results(sim_results, plot_mass=True, plot_fdr=True, plot_fdc=True, plot_branch_distr=True, dla_evolutions=None, title_on=False):
    """
    Unpacks the results of a DLA simulation series, calculates statistics and plots the results.
    inputs:
        sim_results (pd.DataFrame) - a dataframe of simulation results
        plot_mass (bool) - whether to plot the mass distribution for each parameter combination; defaults to True
        plot_fdr (bool) - whether to plot the radius-based fractal dimension for each parameter combination; defaults to True
        plot_fdc (bool) - whether to plot the coarse-graining-based fractal dimension for each parameter combination; defaults to True
        plot_branch_distr (bool) - whether to plot the branch distribution for each parameter combination; defaults to True
        dla_evolutions (dict) - a dictionary of saved DLA evolutions in the form of n-D lattice series
            containing the aggregate states and particle positions over time; defaults to None
        title_on (bool) - whether to display the parameter combination as the title of each plot; defaults to False
    """

    assert isinstance(sim_results, pd.DataFrame), 'sim_results must be a pandas DataFrame'

    sim_results = sim_results.replace(np.nan, None)

    # Unpack simulation parameters
    lattice_size_series = sim_results['lattice_size'].unique()
    max_timesteps_series = sim_results['max_timesteps'].unique()
    seeds_series = sim_results['seeds'].apply(convert_to_tuple).unique()
    particle_density_series = sim_results['particle_density'].unique()
    target_mass_series = sim_results['target_mass'].unique()
    drift_vec_series = sim_results['drift_vec'].apply(tuple).unique()
    sun_vec_series = sim_results['sun_vec'].apply(tuple).unique()
    obstacle_box_series = sim_results['obstacle_boxes'].apply(convert_to_tuple).unique()

    # Reconstruct parameter combinations
    param_combos = product(lattice_size_series, max_timesteps_series, seeds_series, particle_density_series, target_mass_series,
                           drift_vec_series, sun_vec_series, obstacle_box_series)
    
    param_combos_list = list(param_combos)
    n_axes = int(plot_mass) + int(plot_fdr) + int(plot_fdc) + int(plot_branch_distr) + int(dla_evolutions is not None)
    
    # Iterate over parameter combinations
    for i, combo in enumerate(param_combos_list):

        # Set up plot
        if n_axes > 0:
            fig, axs = plt.subplots(1, n_axes)
            fig.set_size_inches(3*n_axes, 2.75)
            str_combo = str(combo).split(',')
            split_idx = int(len(str_combo)*0.5)
            if title_on:
                fig.suptitle(f"Simulation results for\n" + ','.join(str_combo[:split_idx])+ "\n" + ','.join(str_combo[split_idx:]), weight='bold')
        
        # Get data subset for current parameter combination
        param_col = sim_results.columns[:8]
        sim_results['seeds'] = sim_results['seeds'].apply(convert_to_tuple)
        sim_results['drift_vec'] = sim_results['drift_vec'].apply(tuple)
        sim_results['sun_vec'] = sim_results['sun_vec'].apply(tuple)
        sim_results['obstacle_boxes'] = sim_results['obstacle_boxes'].apply(convert_to_tuple)
        data_subset = sim_results[sim_results[param_col].apply(lambda row: np.all([np.array_equal(row[col], val) for col, val in zip(param_col, combo)]), axis=1)]

        # Veriify that subset is not empty
        if data_subset.shape[0] == 0:
            continue

        # Unpack simulation results
        sim_measures = data_subset['sim_measures'].values

        ax_ct = 0

        if dla_evolutions is not None:

            # Turn first axis into a 3D axis
            if dla_evolutions is not None and len(combo[2][0]) == 3:
                axs[ax_ct].remove()
                axs[ax_ct] = fig.add_subplot(n_axes, 1, ax_ct+1, projection='3d')

            # if plot_branch_distr:
            #     branches = [measures['branches'] for measures in sim_measures][0]
            # else:
            #     branches = None
            branches = None

            # Plot DLA evolution
            sample_index = data_subset['evol_ref'].dropna().to_numpy(dtype=int)[0]
            cluster_sample = dla_evolutions['lattice_frames'][sample_index][-1]
            vt.plot_lattice(cluster_sample, branches, ax=axs[ax_ct])
            ax_ct += 1
        
        if plot_mass:
            # Trim mass series to the same length
            min_length = np.min([len(measures['mass_series']) for measures in sim_measures])
            mass_series_trimmed = np.array([np.vstack((np.arange(min_length), measures['mass_series'][:min_length])).T for measures in sim_measures])
            mass_series_trimmed = mass_series_trimmed.reshape(-1, mass_series_trimmed.shape[-1])
            
            # Plot mass over time
            vt.plot_mass_over_time(mass_series_trimmed, ax=axs[ax_ct])
            ax_ct += 1
        
        if plot_fdr:
            # Trim series to the same length
            min_length = np.min([len(measures['fdr_scale_series']) for measures in sim_measures])
            scale_series_trimmed = np.array([measures['fdr_scale_series'][:min_length] for measures in sim_measures])
            n_box_series_trimmed = np.array([measures['fdr_n_box_series'][:min_length] for measures in sim_measures])
            coeffs = np.array([measures['fdr_coeffs'] for measures in sim_measures])
            
            scale_series_trimmed = scale_series_trimmed.flatten()
            n_box_series_trimmed = n_box_series_trimmed.flatten()

            # Perform linear regression on results
            if np.any(np.equal(coeffs, None)):
                coeffs = csm.linreg_fdim(scale_series_trimmed, n_box_series_trimmed)
            else:
                coeffs = np.mean(coeffs, axis=0)

            # Plot fractal dimension (radius scale mode)
            vt.plot_fractal_dimension(scale_series_trimmed, n_box_series_trimmed, coeffs, ax=axs[ax_ct],
                                      title="DLA cluster radius ($s=r$) vs\nnumber of occupied lattice sites ($N(r)$)")
            ax_ct += 1
        
        if plot_fdc:
            scale_series = np.array([measures['fdc_scale_series'] for measures in sim_measures]).flatten()
            n_box_series = np.array([measures['fdc_n_box_series'] for measures in sim_measures]).flatten()
            coeffs = np.array([measures['fdc_coeffs'] for measures in sim_measures])

            # Perform linear regression on results
            if np.any(np.equal(coeffs, None)):
                coeffs = csm.linreg_fdim(scale_series, n_box_series)
            else:
                coeffs = np.mean(coeffs, axis=0)

            # Plot fractal dimension (coarse-graining mode)
            vt.plot_fractal_dimension(scale_series, n_box_series, coeffs, ax=axs[ax_ct],
                                      title="Lattice scaling factor ($s=1/\\epsilon$) vs\nnumber of occupied lattice sites ($N(1/\\epsilon)$)")
            ax_ct += 1

        if plot_branch_distr:
            branch_lengths = [measures['branch_lengths_unique'] for measures in sim_measures]
            branch_length_counts = [measures['branch_length_counts'] for measures in sim_measures]
            branches = [measures['branches'] for measures in sim_measures]

            # Get final mass of each simulation
            max_mass = [measures['mass_series'][-1] for measures in sim_measures]
            assert max_mass is not None, 'mass cannot be None for branch distribution calculation'
            
            # Flatten lists
            branch_lengths_flat = np.array([length for b in branch_lengths for length in b])
            branch_length_ct_flat = np.array([count for b in branch_length_counts for count in b])
            branches_flat = np.array([branch for b in branches for branch in b])

            # Take the average branch length counts over simulations
            branch_lengths_unique = np.unique(branch_lengths_flat)
            branch_length_ct_mean = np.array([np.mean(branch_length_ct_flat[np.argwhere(branch_lengths_flat == length)]) for length in branch_lengths_unique])

            # Plot branch distribution
            vt.plot_branch_length_distribution(branch_lengths_unique, branch_length_ct_mean, branches=branches_flat, ax=axs[ax_ct])
            ax_ct += 1

        plt.tight_layout()
        plt.show()


def analyse_environmental_params(sim_results_param, growth = False, fdr=False, fdc=False, branch=False):
    """
    Analyzes the structural features and 'coralness' of the corals for one value of a changed parameter.
    Four things are analyzed:
    1. Speed of growth, determined by the amount of time steps it takes to get to the target mass.
    2. Fractal Dimension, using the radius to determine the scale.
    3. Fractal Dimension, using coarse graining to determine the scale.
    4. Slope of powerlaw found for the length of the DLA branches.

    input:
        - sim_results_param (np.array): the results of multiple runs of the simulation for the specific value of 
            a parameter.
        - growth (Boolean): whether the speed of growth (1) is given as output.
        - fdr (Boolean): whether the fractal dimension using radius scale (2) is given as output.
        - fdc (Boolean): whether the fractal dimension using coarse graining scale (3) is given as output.
        - branch (Boolean): whether the slope the power law fitted to the branch sizes (4) is given as output.
    
    output:
        - growth_series_mean (float): the average amount of time steps before the target mass is reached.
        - fdr (float): fractal dimension determined with radius scale 
        - fdc (float): fractal dimension determined with coarse-graining scale
        - slope (float): slope of the power law fitted to the branch sizes
    """
    # 1. Speed of growth (time steps needed to reach target mass)
    if growth:
        growth_series = np.array([len(measures['mass_series']) for measures in sim_results_param['sim_measures']])
        growth_series_mean = np.mean(growth_series)
        growth_series_ci = stat.t.interval(0.95, len(growth_series) - 1, growth_series_mean, stat.sem(growth_series))

    # 2. Fractal Dimension with Radius
    if fdr:
        # Trim series to the same length
        min_length = np.min([len(measures['fdr_scale_series']) for measures in sim_results_param['sim_measures']])
        scale_series_trimmed = np.array([measures['fdr_scale_series'][:min_length] for measures in sim_results_param['sim_measures']])
        n_box_series_trimmed = np.array([measures['fdr_n_box_series'][:min_length] for measures in sim_results_param['sim_measures']])
        
        scale_series_trimmed = scale_series_trimmed.flatten()
        n_box_series_trimmed = n_box_series_trimmed.flatten()

        # Perform linear regression on results; use the slope
        fdr = csm.linreg_fdim(scale_series_trimmed, n_box_series_trimmed)[0]

    # 3. Fractal Dimension with Coarse Graining
    if fdc:
        scale_series = np.array([measures['fdc_scale_series'] for measures in sim_results_param['sim_measures']]).flatten()
        n_box_series = np.array([measures['fdc_n_box_series'] for measures in sim_results_param['sim_measures']]).flatten()
        
        # Perform linear regression; use the slope
        fdc = csm.linreg_fdim(scale_series, n_box_series)[0]

    # 4. Branch length distribution fitted with a power law
    if branch:
        branch_lengths = [measures['branch_lengths_unique'] for measures in sim_results_param['sim_measures']]
        branch_length_counts = [measures['branch_length_counts'] for measures in sim_results_param['sim_measures']]

        # Get final mass of each simulation
        max_mass = [measures['mass_series'][-1] for measures in sim_results_param['sim_measures']]
        assert max_mass is not None, 'mass cannot be None for branch distribution calculation'
        
        # Flatten lists
        branch_lengths_flat = np.array([length for bl in branch_lengths for length in bl])
        branch_length_ct_flat = np.array([count for bl in branch_length_counts for count in bl])
        
        # Take the average branch length counts over simulations
        branch_lengths_unique = np.unique(branch_lengths_flat)
        branch_length_ct_mean = np.array([np.mean(branch_length_ct_flat[np.argwhere(branch_lengths_flat == length)]) for length in branch_lengths_unique])
        slope = pl.Fit(branch_length_ct_mean).power_law.alpha

    return growth_series_mean, growth_series_ci, fdr, fdc, slope

def plot_environmental_results(param, sim_results, drift_vec_angle = False, growth = True, fd = True, branch = True):
    """
    Plots the results of changes in the environmental measures.
    inputs:
        - param (string): one of four different environmental parameters (sun, drift angle, drift magnitude, nutrient density)
        - sim_results (pd dataframe): simulation results where the value of param is changed
        - drift_vec_angle (Bool): when the drift vector is changed, this input is used to tell whether the angle is changed (or magnitude)
        - growth (Bool): whether you want to plot the speed of growth output plot
        - fd (Bool): whether you want the fractal dimension plot
        - branch (Bool): whether you want the plot that shows how the powerlaw exponent changes for different param values
    
    """
    assert param == 'drift_vec' or param == 'sun_vec' or param == 'particle_density', 'param must be a string and either drift_vec, sun_vec or particle_density'
    
    # Initialize saving lists 
    mean_time_steps = []
    ci_time_steps = []
    fdr_list = []
    fdc_list = []
    branch_slopes = []

    # Determine parameter values, transform the parameters and set plot settings
    if param == 'sun_vec':
        param_series = sim_results['sun_vec'].apply(tuple).unique()
        # normalize vector
        x_values = [np.linalg.norm(vec) for vec in param_series]
        xlabel = 'Normalized sun vector ($|\\vec{s}|$)'
        x_log = True
    elif param == 'drift_vec':
        param_series = sim_results['drift_vec'].apply(tuple).unique()
        if drift_vec_angle:
            # transform vector to the angle in degrees
            x_values = [math.degrees(math.atan(vec[1]/vec[0])) for vec in param_series]
            xlabel = 'Angle of drift (degrees)'
            x_log = False
        else:
            # normalize vector
            x_values = [np.linalg.norm(vec) for vec in param_series]
            xlabel = 'Norm of drift vector ($|\\vec{d}|$)'
            x_log = True
    elif param == 'particle_density':
        param_series = sim_results['particle_density'].unique()
        x_values = param_series
        xlabel = 'Nutrient density'
        x_log = False

    # Obtain the measures
    for param_value in param_series:
        if param == 'particle_density':
            mask = sim_results[param].apply(lambda x: x == param_value)
            param_filtered = sim_results[mask]
        else: 
            mask = sim_results[param].apply(lambda x: x == list(param_value))
            param_filtered = sim_results[mask]
        mean_time_step, ci_time_step, fdr_value, fdc_value, slope = analyse_environmental_params(param_filtered, growth = growth, fdr= fd, fdc=fd, branch=branch)
    
        mean_time_steps.append(mean_time_step)
        ci_time_steps.append(ci_time_step)
        fdr_list.append(fdr_value)
        fdc_list.append(fdc_value)
        branch_slopes.append(slope)
    
    if growth:
        # Plot Growth
        lower_ci = [interval[0] for interval in ci_time_steps]
        upper_ci = [interval[1] for interval in ci_time_steps]
        plt.fill_between(x_values, lower_ci, upper_ci, color = 'yellowgreen', alpha = 0.5)
        env_param_plot(x_values, mean_time_steps, xlabel, 'Time steps to target mass', color = 'green', title = False, x_log = x_log)
    if fd:
        # Plot Fractal Dimension radius
        plt.plot(x_values, fdr_list, color = 'firebrick', label = 'Radius')
        # Plot Fractal Dimension coarse grain and add plot settings
        env_param_plot(x_values, fdc_list, xlabel, 'Fractal Dimension', color = 'lightcoral', title = False, x_log = x_log, label = 'Coarse grain', legend = True)
    if branch:
        # Plot Branching Slope
        env_param_plot(x_values, branch_slopes, xlabel, 'Powerlaw Exponent', color = 'blue', title = False, x_log = x_log)


def env_param_plot(x_values, y_values, x_label, y_label, color, title = False, x_log = False, label = False, legend = False):
    plt.plot(x_values, y_values, color = color, label = label)
    plt.tick_params(labelsize= 15)
    plt.xlabel(x_label, fontsize = 17)
    plt.ylabel(y_label, fontsize = 17)
    if title:
        plt.title(title, fontsize = 19)
    if x_log:
        plt.semilogx()
    if legend:
        plt.legend(fontsize = 15)
    plt.show()