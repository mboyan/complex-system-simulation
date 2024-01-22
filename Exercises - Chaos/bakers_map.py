import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def animated_scatterplot(pt_data):
    
    assert np.ndim(pt_data) == 3

    n_timesteps = pt_data.shape[0]

    fig, ax = plt.subplots()
    scat = ax.scatter(pt_data[0].T[0], pt_data[0].T[1], c='orange')
    ax.set_aspect('equal', 'box')
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    def update(frame):
        scat.set_offsets(pt_data[frame])
        return scat
    
    anim = animation.FuncAnimation(fig=fig, func=update, frames=n_timesteps, interval=300)
    plt.show()

    # return HTML(anim.to_html5_video())


def baker_iteration(pt):
    """A single iteration of Baker's map.
    input: pt as a (Nx2) numpy array containing
    N points with 2 coordinates.
    """
    pt_new = np.array(pt)
    pt_new[:, 0] = np.where(pt_new[:, 0] < 0.5, 2*pt_new[:, 0], 2-2*pt_new[:, 0])
    pt_new[:, 1] = np.where(pt_new[:, 0] < 0.5, 0.5*pt_new[:, 1], 1-0.5*pt_new[:, 1])

    return pt_new

def main():
    
    pt_density = 200

    # pts_start = np.random.uniform(size=(pt_density, 2))

    x_coords = np.linspace(0, 1, pt_density)
    y_coords = np.linspace(0, 1, pt_density)

    x_coords, y_coords = np.meshgrid(x_coords, y_coords)
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    pts_start = np.vstack((x_coords, y_coords)).T

    n_iter = 20

    pts_current = np.array(pts_start)
    pts_running = np.empty(np.insert(pts_current.shape, 0, n_iter))

    for i in range(n_iter):
        pts_running[i] = np.array(pts_current)
        pts_current = baker_iteration(pts_current)
    
    animated_scatterplot(pts_running)


if __name__ == '__main__':
    main()