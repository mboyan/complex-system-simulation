import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def animated_time_series(data, mu_series):
    
    assert np.ndim(data) == 3

    n_param_steps = data.shape[0]
    n_samples = data.shape[1]

    fig, ax = plt.subplots()
    lineplots = []
    for i in range(n_samples):
        lineplots.append(ax.plot(data[0,:,i].flatten())[0])
    lineplots.append(ax.annotate(mu_series[0], xy=(5, 0.9))) # Put annotation in the same array

    def update(frame):
        for i in range(n_samples):
            lineplots[i].set_ydata(data[frame,:,i].flatten())
        lineplots[-1].set_text(f"$\mu=${mu_series[frame]}")
        return lineplots
    
    anim = animation.FuncAnimation(fig=fig, func=update, frames=n_param_steps, interval=300)
    plt.show()

    # return HTML(anim.to_html5_video())


def tent_iteration(x, mu):
    """A single iteration of the Tent Map
    """
    x_new = np.where(x < 0.5, mu*x, mu*(1-x))
    return x_new


def main():
    
    n_iterations = 10
    n_samples_x = 10
    n_samples_mu = 100

    mu_series = np.linspace(0, 3, n_samples_mu)

    x_evolutions = np.empty((n_samples_mu, n_iterations, n_samples_x))

    for i, mu in enumerate(mu_series):
        x_current = np.linspace(0, 1, n_samples_x)
        for j in range(n_iterations):
            x_evolutions[i,j] = x_current
            x_current = tent_iteration(x_current, mu)

    animated_time_series(x_evolutions, mu_series)


if __name__ == '__main__':
    main()