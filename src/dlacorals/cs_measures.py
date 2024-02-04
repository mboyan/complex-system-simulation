"""
This module contains functions for calculating various complex systems measures.
"""

import numpy as np
import math
import networkx as nx
import powerlaw as pl
from itertools import product

def fractal_dimension_clusters(lattice_array, fit=False):
    """
    Calculate the Minkowski dimension (box-counting method) of a structure on a lattice
    by clustering regions of cells with varying sizes and counting the number of clusters
    containing non-negative cells. Performs a linear regression on the log-log plot of
    number of clusters vs cluster size to determine the fractal dimension.
    inputs:
        lattice_array (np.ndarray) - represents a lattice with equal measure in each dimension (should be a power of 2)
        fit (bool) - whether to perform a linear regression on the log-log plot of number of clusters vs inverse cluster size; defaults to False
    outputs:
        dim_box_series (np.ndarray) - a series of box dimensions
        n_box_series (np.ndarray) - a series of occupied box counts
        scale_series (np.ndarray) - a series of box sizes
        coeffs (np.ndarray) - the coefficients of the linear regression
    """

    lattice_size = lattice_array.shape[0]
    lattice_dims = np.ndim(lattice_array)
    box_exponent = math.log2(lattice_size)

    assert box_exponent % 1.0 == 0, 'lattice size is not a power of 2'

    box_exponent = int(box_exponent)
    dim_box_series = np.empty(box_exponent)
    n_box_series = np.empty(box_exponent)
    scale_series = np.empty(box_exponent)

    # Iterate over different lattice resolutions
    for k in range(1, box_exponent + 1):
        n_boxes = 2 ** k
        box_size = lattice_size // n_boxes

        # Create a series of intervals
        partitions = np.arange(n_boxes + 1) * box_size
        slices_1D = [slice(partitions[i], partitions[i + 1]) for i in range(n_boxes)]

        # Create n-dimensional slices
        slices = tuple(product(slices_1D, repeat=lattice_dims))

        # Count number of boxes including occupied lattice sites
        n_box_occupied = 0
        for s in slices:
            # Extract square region
            lattice_region = lattice_array[s]
            if np.sum(lattice_region) > 0:
                n_box_occupied += 1
        
        # Calculate Minkowski dimension estimate
        dim_box = math.log(n_box_occupied) / math.log(lattice_size/box_size)

        dim_box_series[k - 1] = dim_box
        scale_series[k - 1] = lattice_size / box_size
        n_box_series[k - 1] = n_box_occupied
    

    # Perform linear regression on results
    if fit:
        coeffs = linreg_fdim(scale_series, n_box_series)
    else:
        coeffs = None

    return dim_box_series, scale_series, n_box_series, coeffs


def fractal_dimension_radius(radius_series, n_box_series, fit=False):
    """
    Calculate the Minkowski dimension (box-counting method) of a DLA structure by
    taking the maximum radius of the structure from the initial seed point as a scale reference.
    inputs:
        radius_series (ndarray of float) - a series of maximum radii from the initial seed point, representing different scales
        n_box_series (ndarray of int) - a series of occupied box counts corresponding to the radius series
        fit (bool) - whether to perform a linear regression on the log-log plot of number of clusters vs radius; defaults to False
    outputs:
        dim_box_series (np.ndarray of float) - a series of box dimensions
        coeffs (np.ndarray of float) - the coefficients of the linear regression
    """
    assert radius_series.shape == n_box_series.shape, 'radius_series and n_box_series must have the same shape'

    # Calculate Minkowski dimension estimate
    dim_box_series = np.log(n_box_series) / np.log(radius_series)

    # Perform linear regression
    if fit:
        coeffs = linreg_fdim(radius_series, n_box_series)
    else:
        coeffs = None

    return dim_box_series, coeffs


def branch_distribution(lattice_array, seed_coords, moore=False, verify_plaw=False):
    """
    Computes the branch distribution of a DLA structure by tracing the shortest paths
    from the branch tips to the seed point, identifying the membership of a lattice site
    to a specific branch by the number of paths that pass through it and counting the number
    of branches of different lengths.
    inputs:
        lattice_array (np.ndarray) - a lattice grid where non-zero values represent occupied lattice sites
        seed_coords (tuple) - the coordinates of the seed point
        moore (bool) - whether to use the Moore neighborhood (8-connected) or the Von Neumann neighborhood (4-connected); default is Von Neumann
        verify_plaw (bool) - when True, verifies whether the branch length distribution follows a power law distribution; default is False
    outputs:
        branch_lengths_unique (np.ndarray) - an array of unique branch lengths
        branch_length_counts (np.ndarray) - an array of corresponding branch length counts
        branches (list) - a list of lists of nodes (tuples) representing the branches of a DLA cluster
    """
    
    seed_coords = tuple(seed_coords)
    
    assert lattice_array[seed_coords] == 1, 'seed point must be occupied'

    lattice_dims = np.ndim(lattice_array)

    # Create an empty graph
    G = nx.Graph()

    # Create a network over the non-zero lattice sites
    for index in np.ndindex(lattice_array.shape):

        # If the current cell is occupied
        if lattice_array[index] > 0:

            # Add a node for the current cell
            G.add_node(index)

            # Define Moore neighbourhood
            offsets = [step for step in product([0, 1, -1], repeat=lattice_dims) if np.linalg.norm(step) != 0]
            offsets = np.array(offsets)

            # Reduce to von Neumann neighbourhood
            if not moore:
                offsets = offsets[np.sum(np.abs(offsets), axis=1) == 1]

            # Connect the current cell to its neighbours
            for offset in offsets:
                # Skip the current cell
                if all(o == 0 for o in offset):
                    continue
                # Compute the neighbor's coordinates
                neighbour = tuple(i + o for i, o in zip(index, offset))
                # If the neighbor is within the lattice and its value is 1
                if all(0 <= i < s for i, s in zip(neighbour, lattice_array.shape)) and lattice_array[neighbour] == 1:
                    # Add an edge from the current cell to the neighbor
                    G.add_edge(index, neighbour)

    # Plot the graph with coordinates
    # pos = {node: node for node in G.nodes()}
    # nx.draw(G, pos=pos, node_size=1, node_color='k', edge_color='k', width=0.5)

    # Trace all paths to seed
    paths = [nx.shortest_path(G, node, seed_coords) for node, degree in G.degree()]

    # Find non-overlapping nodes
    path_sets = [set(path) for path in paths]
    branches = []
    for _ in range(len(path_sets)):

        # Sort paths by length
        path_sets.sort(key=len)

        # Take the longest path
        longest_path = path_sets.pop()

        if len(longest_path) == 0:
            break

        # Remove nodes from the longest path from the remaining paths
        path_sets = [path_set - longest_path for path_set in path_sets]

        longest_path_list = list(longest_path)
        branches.append(longest_path_list)

    # Count the number of branches of different lengths
    branch_lengths = [len(branch) for branch in branches]
    branch_lengths_unique = np.unique(branch_lengths)
    branch_length_counts = [branch_lengths.count(length) for length in branch_lengths_unique]

    if verify_plaw:
        verify_power_law(branch_lengths)

    return branch_lengths_unique, branch_length_counts, branches


def linreg_fdim(scale_series, n_box_series):
    """Perform linear regression on the log-log plot of the fractal dimension results
    inputs:
        scale_series (np.ndarray) - a series of scale factors
        n_box_series (np.ndarray) - a series of occupied box counts
    outputs:
        coeffs (np.ndarray) - the coefficients of the linear regression
    """
    assert scale_series.shape == n_box_series.shape, 'scale_series and n_box_series must have the same shape'

    log_radius_series = np.log(scale_series)
    log_n_box_series = np.log(n_box_series)
    coeffs = np.polyfit(log_radius_series, log_n_box_series, 1)

    return coeffs


def verify_power_law(data):
    """
    Verifies whether a dataset follows a power law distribution by fitting a power law function
    and comparing it to an exponential one.
    inputs:
        data (list) - a list of data points
    outputs:
        loglikelihood (float) - the log-likelihood of the power law fit
        p_value (float) - the p-value of the power law fit
    """
    powerlaw_results = pl.Fit(data, discrete=True)
    loglikelihood, p_value = powerlaw_results.loglikelihood_ratio('power_law', 'exponential')
    plaw_verification = f'Plaw-vs-exp-likelihood:\n{loglikelihood:.4f}, p = {p_value:.4f}'
    alpha = powerlaw_results.power_law.alpha
        
    return loglikelihood, p_value, alpha, plaw_verification