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

    # Iterate over different lattice resolutions
    for k in range(1, box_exponent + 1):
        box_size = lattice_size // k

        print(box_size)

        # Count number of boxes including occupied lattice sites

        # Create a series of intervals
        intervals = [((i * box_size, (i + 1) * box_size - 1) for i in range(k))]
        intervals_product = list(product(intervals, repeat=lattice_dims))

        # Create n-dimensional slices
        slices = [slice(ip[0], ip[1]) for ip in intervals_product]
        lattice_regions = lattice_array[tuple(slices)]

        print(lattice_regions.shape)
        # region_sums = np.sum(lattice_regions)
        # n_box_occupied = asd

        # n_box_occupied = 0
        # for i in range(k):

        #     # Create a list of slice objects
        #     slices = [slice(i * box_size, (i + 1) * box_size - 1) for _ in range(lattice_dims)]

        #     # Extract square region
        #     lattice_region = lattice_array[tuple(slices)]
        #     if np.sum(lattice_region) > 0:
        #         n_box_occupied += 1
        
        # # Calculate Minkowski dimension estimate
        # dim_box = math.log(n_box_occupied) / math.log(box_size)

        # dim_box_series[k - 1] = dim_box
    
    # return dim_box_series