"""
This module contains functions for visualising plots and animations of DLA simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib import animation
from IPython.display import HTML

from . import cs_measures as csm

def plot_lattice_2D(lattice, cmap='tab20b', ax=None, title=None, cell_size=1):
    """
    Plots a 2D lattice
    inputs:
        lattice (np.ndarray) - a 2D lattice array
        cmap (str) - a matplotlib colormap; defaults to 'tab20b'
        ax (matplotlib.axes.Axes) - an axis to plot on; defaults to None
        title (str) - a title for the plot; defaults to None
    """

    assert np.ndim(lattice) == 2, 'input array must have 2 dimensions'

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(lattice.shape[0]*0.05*cell_size, lattice.shape[1]*0.05*cell_size)
    else:
        fig = ax.get_figure()
    
    # Align with plot orientation
    lattice = np.moveaxis(lattice, 0, 1)
    lattice = np.flip(lattice, axis=0)

    ax.imshow(lattice, cmap=cmap)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if title is not None:
        ax.set_title(title, fontsize='small')
    else:
        ax.set_title("DLA cluster", fontsize='small')

    plt.axis('off')

    if ax is None:
        plt.show()


def plot_lattice_3D(lattice, cmap='tab20b', ax=None, title=None):
    """
    Plots a 3D lattice
    inputs:
        lattice (np.ndarray) - a 3D lattice array
        cmap (str) - a matplotlib colormap; defaults to 'tab20b'
        ax (matplotlib.axes.Axes) - an axis to plot on; defaults to None
        title (str) - a title for the plot; defaults to None
    """

    assert np.ndim(lattice) == 3, 'input array must have 3 dimensions'

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    else:
        fig = ax.get_figure()

    # Create a colormap that maps integers to colors
    cmap = cm.get_cmap(cmap)

    # Normalize lattice data
    norm = lattice / np.max(lattice)

    # Create a 3D plot where each voxel is colored based on its value
    cols = cmap(norm)

    # Set alpha channel
    cols[..., -1] = norm*0.75

    ax.voxels(lattice, facecolors=cols, linewidth=0, alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if title is not None:
        ax.set_title(title, fontsize='small')
    else:
        ax.set_title("DLA cluster", fontsize='small')

    plt.axis('off')

    if ax is None:
        plt.show()


def plot_lattice(lattice, branches=None, ax=None, title=None, cell_size=1):
    """
    Check lattice data and dispatch to either 2D or 3D plot function.
    inputs:
        lattice (np.ndarray) - a 2D or 3D lattice array
        branches (list) - a list of lists of nodes (tuples) representing the branches of a DLA cluster;
            if provided, the branches will be assigned different colours; defaults to None
        ax (matplotlib.axes.Axes) - an axis to plot on; defaults to None
        title (str) - a title for the plot; defaults to None
    """
    assert len(set(lattice.shape)) == 1, 'lattice is not a square array'

    if branches is None:
        branch_lattice = np.array(lattice, dtype=np.float64)
        cmap = 'tab20b'
    else:
        # Assign colours based on branches
        branch_lattice = np.full(lattice.shape, np.nan)
        for i, branch in enumerate(branches):
            for node in branch:
                branch_lattice[node] = i + 1
        cmap = 'prism'
    
    if np.ndim(lattice) == 2:
        plot_lattice_2D(branch_lattice, cmap, ax, title, cell_size)
    elif np.ndim(lattice) == 3:
        plot_lattice_3D(branch_lattice, cmap, ax, title)
    else:
        raise ValueError('input array must have 2 or 3 dimensions')


def animate_lattice_2D(lattice_data_frames, interval=100):
    """
    Creates a 2D plot animation from snapshots of lattice states.
    inputs:
        lattice_data_frames (3D numpy.ndarray) - time frames of lattice states
        interval (int) - time between frames in milliseconds
    outputs:
        an HTML5 video of the animation
    """

    assert np.ndim(lattice_data_frames) == 3, 'error in input array dimensions'

    n_frames = lattice_data_frames.shape[0]

    cell_size = 2

    # Flip lattice data frames to match the orientation of the animation
    lattice_data_frames = np.moveaxis(lattice_data_frames, 1, 2)
    lattice_data_frames = np.flip(lattice_data_frames, axis=1)

    # Set up figure and axis
    fig = plt.figure()
    # fig = plt.figure(figsize=(lattice_data_frames.shape[1] * cell_size, lattice_data_frames.shape[2] * cell_size))
    img = plt.imshow(np.random.randint(2, size=((lattice_data_frames.shape[1] * cell_size, lattice_data_frames.shape[2] * cell_size))), cmap = 'tab20b')

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
    outputs:
        an HTML5 video of the animation
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
    outputs:
        an HTML5 video of the animation
    """
    
    if np.ndim(lattice_data_frames) == 3:
        anim = animate_lattice_2D(lattice_data_frames, interval)
    elif np.ndim(lattice_data_frames) == 4:
        anim = animate_lattice_3D(lattice_data_frames, interval)
    else:
        raise ValueError('input array must have 3 or 4 dimensions')
    
    plt.show()
    return anim


def plot_fractal_dimension(scale_series, n_box_series, coeffs, ax=None, title=None):
    """
    Plots the relationship between N(epsilon) (number of boxes of size epsilon)
    and 1/epsilon (inverse of box size) on a log-log plot, illustrating the analysed fractal dimension
    inputs:
        scale_series (np.ndarray) - a series of box sizes
        n_box_series (np.ndarray) - a series of occupied box counts
        coeffs (np.ndarray) - the coefficients of the linear regression
        ax (matplotlib.axes.Axes) - an axis to plot on; defaults to None
        title (str) - a title for the plot; defaults to None
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    ax.scatter(scale_series, n_box_series, marker='.', label=f'scale-mass relationship')
    ax.set_xscale('log')
    ax.set_yscale('log')
    log_scale_series = np.log(scale_series)
    log_n_boxes_fit = coeffs[0] * log_scale_series + coeffs[1]
    n_boxes_fit = np.exp(log_n_boxes_fit)
    sort_indices = np.argsort(scale_series)
    ax.loglog(scale_series[sort_indices], n_boxes_fit[sort_indices], linestyle='--', color='red', label=f'regression ($D={coeffs[0]:.5f}$)')
    
    ax.set_xlabel("$s$")
    ax.set_ylabel("$N(s)$")
    ax.legend(fontsize='small')

    if title is None:
        ax.set_title("Lattice scaling factor (s) vs\nnumber of occupied lattice sites (N)", fontsize='small')
    else:
        ax.set_title(title, fontsize='small')

    if ax is None:
        plt.show()


def plot_branch_length_distribution(branch_lengths_unique, branch_length_counts, branches=None, ax=None, title=None):
    """
    Plots the distribution of branch lengths in a DLA cluster
    inputs:
        branch_lengths_unique (list) - a list of unique branch lengths
        branch_length_counts (list) - a list of the number of branches of each length
        branches (list) - a list of lists of nodes (tuples) representing the branches of a DLA cluster; if provided,
            the power law distribution will be verified; defaults to None
        ax (matplotlib.axes.Axes) - an axis to plot on; defaults to None
        title (str) - a label for the plot; defaults to None
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.scatter(branch_lengths_unique, branch_length_counts, marker='.')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Branch mass ($L$)")
    ax.set_ylabel("$N(L)$")
    ax.legend()

    if title is None:
        ax.set_title("Distribution of branch lengths in a DLA cluster", fontsize='small')
    else:
        ax.set_title(title, fontsize='small')

    if branches is not None:
        branch_lengths = [len(branch) for branch in branches]
        _, _, alpha, plaw_verification = csm.verify_power_law(branch_lengths)

        C = branch_length_counts[2]/(branch_lengths_unique[2]**(-alpha))
        xs = branch_lengths_unique
        ys = C*xs**(-alpha)

        label = f'alpha={alpha:.5f}\n' + plaw_verification
        ax.loglog(xs, ys, color='r', linestyle='--', label=label)
        ax.set_ylim(0.75, np.max(branch_lengths) * 2)
        ax.legend(fontsize='small')

    if ax is None:
        plt.show()


def plot_mass_over_time(mass_series, ax=None, title=None):
    """
    Plots the mass of a DLA cluster over time
    inputs:
        mass_series (ndarray) - an array of tuples (t, m) indicating the time indices and masses at each time step
    """
    
    assert np.ndim(mass_series) == 2, 'input array must have 2 dimensions'

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    sns.lineplot(x=mass_series[:, 0], y=mass_series[:, 1], ax=ax, legend=False)

    ax.set_xlabel("Time")
    ax.set_ylabel("Mass")

    if title is None:
        ax.set_title("Mass of a DLA cluster over time", fontsize='small')
    else:
        ax.set_title(title)

    if ax is None:
        plt.show()