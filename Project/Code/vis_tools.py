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
    img = plt.imshow(np.random.randint(2, size=((lattice_data_frames.shape[1], lattice_data_frames.shape[2]))), cmap = 'bwr')

    # Animation update function
    def animate(i):
        img.set_array(lattice_data_frames[i])
        return img,

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=interval, blit=True)
    plt.axis('off')

    return HTML(anim.to_html5_video())