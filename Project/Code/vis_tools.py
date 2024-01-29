"""
This module contains functions for visualising plots and animations of DLA simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def animate_lattice_2D(lattice_data_frames, interval=100):
    """
    Creates a 2D plot animation from a snapshot of lattice states.
    inputs:
        lattice_data_frames (3D numpy.ndarray) - time frames of lattice states
        interval (int) - time between frames in milliseconds
    """

    assert np.ndim(lattice_data_frames) == 3, 'error in input array domains'

    n_frames = lattice_data_frames.shape[0]

    # Set up figure and axis
    fig = plt.figure()
    # fig = plt.figure(figsize=(lattice_data_frames.shape[1] * cell_size, lattice_data_frames.shape[2] * cell_size))
    img = plt.imshow(np.random.randint(2, size=((lattice_data_frames.shape[1], lattice_data_frames.shape[2]))), cmap = 'tab20b')

    # Animation update function
    def animate(i):
        img.set_array(lattice_data_frames[i])
        return img,

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=interval, blit=True)
    plt.axis('off')

    return HTML(anim.to_html5_video())


def plot_fractal_dimension(scale_series, n_box_series, coeffs):
    """
    Plots the relationship between N(epsilon) (number of boxes of size epsilon)
    and 1/epsilon (inverse of box size) on a log-log plot, illustrating the analysed fractal dimension
    """

    fig, ax = plt.subplots()

    ax.loglog(scale_series, n_box_series, marker='o', label='scale-mass relationship')
    log_scale_series = np.log(scale_series)
    log_n_boxes_fit = coeffs[0] * log_scale_series + coeffs[1]
    n_boxes_fit = np.exp(log_n_boxes_fit)
    ax.loglog(scale_series, n_boxes_fit, linestyle='--', color='red', label=f'regression ($D={coeffs[0]}$)')

    ax.set_xlabel("$1/\\epsilon$")
    ax.set_ylabel("$N(\\epsilon)$")
    ax.legend()
    fig.suptitle("Lattice scaling factor vs number of occupied lattice sites")

    plt.show()