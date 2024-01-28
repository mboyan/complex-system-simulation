"""
This module contains functions for visualising plots and animations of DLA simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
from IPython.display import HTML

def animate_lattice_2D(lattice_data_frames, interval=100):
    """
    Creates a 2D plot animation from snapshots of lattice states.
    inputs:
        lattice_data_frames (3D numpy.ndarray) - time frames of lattice states
        interval (int) - time between frames in milliseconds
    """

    assert np.ndim(lattice_data_frames) == 3, 'error in input array dimensions'

    n_frames = lattice_data_frames.shape[0]

    # Flip lattice data frames to match the orientation of the animation
    lattice_data_frames = np.moveaxis(lattice_data_frames, 1, 2)
    lattice_data_frames = np.flip(lattice_data_frames, axis=1)

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


def animate_lattice_3D(lattice_data_frames, interval=100):
    """
    Creates a 3D plot animation from snapshots of lattice states.
    inputs:
        lattice_data_frames (3D numpy.ndarray) - time frames of lattice states
        interval (int) - time between frames in milliseconds
    """

    assert np.ndim(lattice_data_frames) == 4, 'error in input array dimensions'

    n_frames = lattice_data_frames.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(lattice_data_frames[0], facecolors='r', linewidth=0, alpha=0.6)

    # Animation update function
    def animate(i):
        ax.clear()

         # Create a colormap that maps integers to colors
        cmap = cm.get_cmap('tab20b')

        # Normalize lattice data
        norm = lattice_data_frames[i] / np.max(lattice_data_frames)

        # Create a 3D plot where each voxel is colored based on its value
        cols = cmap(norm)

        # Set alpha channel
        cols[..., -1] = norm*0.75

        ax.voxels(lattice_data_frames[i], facecolors=cols, linewidth=0, alpha=0.6)
        ax.view_init(elev=30, azim=i)

        # Turn off the ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        return [ax]

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=interval, blit=False)
    plt.axis('off')

    return HTML(anim.to_html5_video())


def animate_lattice(lattice_data_frames, interval):
    """
    Check lattice data and dispatch to either 2D or 3D animation function.
    inputs:
        lattice_data_frames (3D numpy.ndarray) - time frames of lattice states
        interval (int) - time between frames in milliseconds
    """
    
    if np.ndim(lattice_data_frames) == 3:
        anim = animate_lattice_2D(lattice_data_frames, interval)
    elif np.ndim(lattice_data_frames) == 4:
        anim = animate_lattice_3D(lattice_data_frames, interval)
    else:
        raise ValueError('input array must have 3 or 4 dimensions')
    
    plt.show()
    return anim


def plot_fractal_dimension(scale_series, n_box_series, coeffs, ax=None, label=None):
    """
    Plots the relationship between N(epsilon) (number of boxes of size epsilon)
    and 1/epsilon (inverse of box size) on a log-log plot, illustrating the analysed fractal dimension
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.loglog(scale_series, n_box_series, marker='o', label=f'scale-mass relationship{label}')
    log_scale_series = np.log(scale_series)
    log_n_boxes_fit = coeffs[0] * log_scale_series + coeffs[1]
    n_boxes_fit = np.exp(log_n_boxes_fit)
    ax.loglog(scale_series, n_boxes_fit, linestyle='--', color='red', label=f'regression ($D={coeffs[0]}$)')

    ax.set_xlabel("$1/\epsilon$")
    ax.set_ylabel("$N(\epsilon)$")
    ax.legend()
    fig.suptitle("Lattice scaling factor vs number of occupied lattice sites")

    if ax is None:
        plt.show()