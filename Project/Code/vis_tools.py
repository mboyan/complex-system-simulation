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
    img = plt.imshow(np.random.randint(2, size=((lattice_data_frames.shape[1], lattice_data_frames.shape[2]))))

    # Animation update function
    def animate(i):
        img.set_array(lattice_data_frames[i])
        return img,

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=interval, blit=True)

    return HTML(anim.to_html5_video())


def plot_fractal_dimension(n_box_series, scale_inverse_series, coeffs):
    """
    Plots the relationship between N(epsilon) (number of boxes of size epsilon)
    and 1/epsilon (inverse of box size) on a log-log plot, illustrating the analysed fractal dimension
    """

    fig, ax = plt.subplots()

    ax.loglog(n_box_series, scale_inverse_series, marker='o')
    coeffs = np.exp(coeffs)
    plot_min = np.min(n_box_series)
    plot_max = np.max(n_box_series)
    ax.plot([plot_min, plot_max], [plot_min * coeffs[0] + coeffs[1], plot_max * coeffs[0] + coeffs[1]])

    ax.set_xlabel("$1/\epsilon$")
    ax.set_ylabel("$N(\epsilon)$")
    fig.suptitle("Scaling factor of lattice vs number of occupied lattice sites")

    plt.show()