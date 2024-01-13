import numpy as np
from scipy.spatial.distance import cdist
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from itertools import product

def show_state_over_time_2D(states_over_time):
    '''
    Function provided by Rick Quax.
    @param states_over_time: a TxLxL array for T time steps and a system size of LxL.
    '''
    assert np.ndim(states_over_time) == 3
    
    num_time_steps = np.shape(states_over_time)[0]
    
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    im = plt.imshow(np.random.randint(3, size=(states_over_time.shape[1], states_over_time.shape[2])), animated=True)
    
    # animation function. This is called sequentially
    def animate(i):
        im.set_array(states_over_time[i])
        return im,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=num_time_steps, interval=200, blit=True)

    plt.show()

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    return HTML(anim.to_html5_video())
#     rc('animation', html='html5')
#     return anim


def show_state_over_time_3D(states_over_time):
    '''
    Based on function provided by Rick Quax.
    @param states_over_time: a TxLxL array for T time steps and a system size of LxL.
    '''
    assert np.ndim(states_over_time) == 4
    
    num_time_steps = np.shape(states_over_time)[0]
    
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplot(projection='3d')
    voxels = ax.voxels(states_over_time[0], facecolor=states_over_time[0], edgecolors=states_over_time[0])
    # im = plt.imshow(np.random.randint(3, size=(states_over_time.shape[1], states_over_time.shape[2])), animated=True)
    
    # animation function. This is called sequentially
    def animate(i):
        # im.set_array(states_over_time[i])
        ax.cla()
        voxels = ax.voxels(states_over_time[i], facecolor=states_over_time[i], edgecolors=states_over_time[i])
        return voxels,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=num_time_steps, interval=200, blit=True)

    plt.show()

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    return HTML(anim.to_html5_video())


def construct_ca_rule_random(n_dims, n_states, langton_lambda=0.5):
    """Construct a random CA rule based on a number of dimensions (n_dims),
    a number of possible states (n_states) and a lambda parameter (langton_lambda).
    Return the rule as a numpy array of integers.
    Assumes a Moore neighbourhood by default
    """

    neighbourhood_size = 3 ** n_dims
    n_alpha = n_states ** neighbourhood_size # possible neighbourhood states

    # Define input alphabet
    rule_alpha = product(range(n_states), repeat=neighbourhood_size)
    rule_alpha = np.array(list(rule_alpha))

    # Combine kernels into integers
    # rule_alpha_integers = np.empty(rule_alpha.shape[0], dtype=int)
    # for i, input in enumerate(rule_alpha):
    #     rule_alpha_integers[i] = np.sum(input * (10 ** np.arange(rule_alpha.shape[1])), dtype=int)

    # Determine number of rules with a quiescent state (s_q = 0)
    n_quiescent = (1 - langton_lambda) * n_alpha

    # Determine the indices of rules leading to non-quescent states
    nonquiescent_indices = np.random.choice(np.arange(n_alpha), int(n_alpha - n_quiescent), replace=False)

    # Fill set of rules
    rule_states = np.zeros(n_alpha, dtype=int)
    rule_states[nonquiescent_indices] = np.random.randint(1, n_states, int(n_alpha - n_quiescent))
    
    # rule_states = np.random.randint(0, n_states, np.power(n_states, neighbourhood_size))

    return rule_alpha, rule_states


def run_ca(rule_array, n_steps, n_dims, n_states, space_size, return_steps=False, return_transient=True):
    """Run CA for a set number of steps (n_steps) over a grid of a set size (space_size ^ n_dims)
    using a rule encoded in a list (rule_array) containing as a first element a numpy array
    of the rule input alphabet and as a second element a numpy array of the rule output states.
    """

    rule_alpha = rule_array[0]
    rule_states = rule_array[1]

    lattice_dims = np.full(n_dims, space_size)
    config_dims = np.insert(lattice_dims, 0, n_steps)
    
    step_configs = np.empty(config_dims, dtype=int)

    # Determine dimension shifts
    dim_shifts = product(range(-1, 2), repeat=n_dims)
    dim_shifts = np.array(list(dim_shifts))
    
    # Initialise random states
    step_configs[0] = np.random.randint(0, n_states, lattice_dims)
    # print(step_configs[0])

    for i in range(1, n_steps):
        
        # Create shifted lattices
        shifted_lattices = np.zeros(np.insert(lattice_dims, 0, dim_shifts.shape[0]), dtype=int)
        for j, ds in enumerate(dim_shifts):
            shifted_lattices[j] = np.roll(step_configs[i-1], ds, np.arange(n_dims))

        # Transpose shifted lattices
        neighbourhoods = np.transpose(shifted_lattices, np.insert(np.arange(shifted_lattices.ndim)[1:], shifted_lattices.ndim-1, 0))

        # Find closest rule alphabet
        neighbourhoods_reshaped = neighbourhoods.reshape((-1, neighbourhoods.shape[-1]))
        print(neighbourhoods_reshaped.shape)
        print(rule_alpha.shape)
        rule_dist = cdist(neighbourhoods_reshaped, rule_alpha)
        closest_index = np.argmin(rule_dist, axis=1)
        closest_rule = rule_states[closest_index]

        # Determine next configuration of states
        step_configs[i] = closest_rule.reshape(step_configs[i].shape)

    return step_configs


def main():

    # CA parameters
    n_steps = 2
    n_dims = 3
    n_states = 2
    space_size = 6
    show_evolution = True
    
    ca_rule = construct_ca_rule_random(n_dims, n_states, 0.25)
    print(ca_rule)

    ca_steps = run_ca(ca_rule, n_steps, n_dims, n_states, space_size)
    # print(ca_steps[1])

    if show_evolution:
        if n_dims == 1:
            # Plot 1D CA with imshow()
            plt.imshow(ca_steps)
            plt.show()
        elif n_dims == 2:
            # Visualise 2D CA animation
            show_state_over_time_2D(ca_steps)
        elif n_dims == 3:
            # Visualise 3D CA animation
            pass


if __name__ == '__main__':
    main()