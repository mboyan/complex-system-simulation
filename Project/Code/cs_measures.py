import numpy as np
import math
from itertools import product

def fractal_dimension(lattice_array):
    """
    Calculate the Minowski dimension (box-counting method) of a structure on a lattice.
    inputs:
        img_array (np.ndarray) - represents a lattice with equal measure in each dimension (should be a power of 2)
    """

    lattice_size = lattice_array.shape[0]
    lattice_dims = np.ndim(lattice_array)
    box_exponent = math.log2(lattice_size)

    assert box_exponent % 1.0 == 0, 'lattice size is not a power of 2'

    box_exponent = int(box_exponent)
    dim_box_series = np.empty(box_exponent)
    n_box_series = np.empty(box_exponent)
    scale_inverse_series = np.empty(box_exponent)

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
        
        # # Calculate Minkowski dimension estimate
        dim_box = math.log(n_box_occupied) / math.log(lattice_size/box_size)

        dim_box_series[k - 1] = dim_box
        n_box_series[k - 1] = n_box_occupied
        scale_inverse_series[k - 1] = lattice_size/box_size
    

    # Perform least-squares regression on results
    log_n_box_series = np.log(n_box_series)
    log_scale_inverse_series = np.log(scale_inverse_series)
    # print(log_n_box_series[:, np.newaxis])
    # reg_solution = np.linalg.lstsq(log_n_box_series[:, np.newaxis], log_scale_inverse_series)
    # print(reg_solution)

    coeffs = np.polyfit(log_scale_inverse_series, log_n_box_series, 1)
    print(coeffs)

    return dim_box_series, n_box_series, scale_inverse_series, coeffs