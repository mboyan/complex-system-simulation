"""
This module contains functions for calculating various complex systems measures.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product

def fractal_dimension_clusters(lattice_array):
    """
    Calculate the Minkowski dimension (box-counting method) of a structure on a lattice
    by clustering regions of cells with varying sizes and counting the number of clusters
    containing non-negative cells. Performs a linear regression on the log-log plot of
    number of clusters vs cluster size to determine the fractal dimension.
    inputs:
        lattice_array (np.ndarray) - represents a lattice with equal measure in each dimension (should be a power of 2)
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
    log_scale_series = np.log(scale_series)
    log_n_box_series = np.log(n_box_series)
    coeffs = np.polyfit(log_scale_series, log_n_box_series, 1)

    return dim_box_series, scale_series, n_box_series, coeffs


def fractal_dimension_radius(radius_series, n_box_series):
    """
    Calculate the Minkowski dimension (box-counting method) of a DLA structure by
    taking the maximum radius of the structure from the initial seed point as a scale reference.
    inputs:
        radius_series - a series of maximum radii from the initial seed point, representing different scales
        n_box_series - a series of occupied box counts corresponding to the radius series
    """
    assert radius_series.shape == n_box_series.shape, 'radius_series and n_box_series must have the same shape'

    # Calculate Minkowski dimension estimate
    dim_box_series = np.log(n_box_series) / np.log(radius_series)

    # Perform linear regression
    # for r in radius_series: print(r)
    # print(n_box_series)
    log_radius_series = np.log(radius_series)
    log_n_box_series = np.log(n_box_series)
    coeffs = np.polyfit(log_radius_series, log_n_box_series, 1)

    return dim_box_series, coeffs


def branch_distribution(lattice_array, seed_coords, moore=False):
    """
    Computes the branch distribution of a DLA structure by tracing the shortest paths
    from the branch tips to the seed point, identifying the membership of a lattice site
    to a specific branch by the number of paths that pass through it and counting the number
    of branches of different lengths.
    inputs:
        lattice_array (np.ndarray) - a lattice grid where non-zero values represent occupied lattice sites
        seed_coords (tuple) - the coordinates of the seed point
        moore (bool) - whether to use the Moore neighborhood (8-connected) or the Von Neumann neighborhood (4-connected); default is Von Neumann
    """
    assert lattice_array[tuple(seed_coords)] == 1, 'seed point must be occupied'

    lattice_dims = np.ndim(lattice_array)

    # Create an empty graph
    G = nx.Graph()

    # Create a network over the non-zero lattice sites
    for index in np.ndindex(lattice_array.shape):

        # Add a node for the current cell
        G.add_node(index)

        # If the current cell is occupied
        if lattice_array[index] > 0:

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

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Draw the graph
    pos = {node: node for node in G.nodes}
    nx.draw(G, pos, ax=ax, node_size=10, node_color='red')

    return