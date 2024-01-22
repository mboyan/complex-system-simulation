import numpy as np
import math
from scipy.spatial.distance import cdist
import multiprocessing as mp

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors
# import mpl_toolkits

from IPython.display import HTML
from itertools import product

class RandomIntStream:
    """A class which generates a random stream of integers of size max_length
    and switches to a different seed if the stream is queried with an out-of-bounds index
    """
    def __init__(self, low, high, max_length, init_seed=0):
        self.low = low
        self.high = high
        # self.generated_ints = self.rng.integers(self.low, self.high, size=max_length)
        self.n_max = max_length
        self.seed = init_seed

    def get(self, n):
        self.seed = int(math.floor(n / self.n_max))
        self.rng = np.random.default_rng(self.seed)
        self.generated_ints = self.rng.integers(self.low, self.high, size=int(n%self.n_max))
        return self.generated_ints[-1]
        # if n < self.n_max:
        #     return self.generated_ints[n]
        # else:
        #     # ARRAY BELOW GETS TOO BIG FOR MEMORY!!!!!
        #     self.generated_ints = np.concatenate((self.generated_ints, self.rng.integers(self.low, self.high, size=n - self.n_max + 1)))
        #     self.n_max = n
        #     return self.generated_ints[n]


# def unique_random_ints(n, int_max):
#     """Returns a numpy array of random non-repeating integers smaller than int_max
#     """
#     rng = np.random.default_rng()
#     unique_ints = np.array([], dtype=int)
#     while unique_ints.size < n:
#         new_int = rng.integers(int_max)
#         if new_int not in unique_ints:
#             unique_ints = np.append(unique_ints, new_int)
#     return unique_ints
# def unique_random_ints(n, int_max):
#     """Returns a numpy array of random non-repeating integers smaller than int_max"""
#     rng = np.random.default_rng()
#     unique_ints_set = set()
#     while len(unique_ints_set) < n:
#         new_ints = set(rng.integers(0, int_max, int_max))
#         #unique_ints_set.add(new_int)
#         unique_ints_set = unique_ints_set | new_ints
#         print(int_max - len(unique_ints_set))
#     return np.array(list(unique_ints_set), dtype=int)


def unique_random_ints(n, int_max):
    """Returns a numpy array of random non-repeating integers smaller than int_max"""
    rng = np.random.default_rng()
    unique_ints_set = set()
    batch_size = min(n, 10000)  # Adjust this value based on your available memory
    while len(unique_ints_set) < n:
        new_ints = set(rng.integers(0, int_max, batch_size))
        unique_ints_set |= new_ints
        print(n - len(unique_ints_set))
    return np.array(list(unique_ints_set)[:n], dtype=int)


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


def show_state_over_time_3D(states_over_time, n_states):
    '''
    Based on function provided by Rick Quax.
    @param states_over_time: a TxLxL array for T time steps and a system size of LxL.
    '''
    assert np.ndim(states_over_time) == 4
    
    num_time_steps = np.shape(states_over_time)[0]

    # Define array of colors
    # norm = plt.Normalize(0, n_states - 1)
    # scalar_map = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('viridis'))
    # colors = scalar_map.to_rgba(np.arange(n_states))
    
    # First set up the figure, the axis, and the plot element we want to animate
    ax = plt.subplot(projection='3d')
    fig = ax.get_figure()
    voxels = ax.voxels(states_over_time[0])
    
    def animate(i):
        ax.clear()  # clear current plot
        # Create a colormap that maps integers to colors
        cmap = plt.get_cmap('viridis', n_states)
        # Create a normalization object that scales data values to the [0, 1] interval
        norm = colors.Normalize(vmin=0, vmax=n_states-1)
        # Create a 3D plot where each voxel is colored based on its value
        voxels = states_over_time[i] > 0
        cols = cmap(norm(states_over_time[i]))
        ax.voxels(voxels, facecolors=cols, edgecolor='k', linewidth=0.5, alpha=0.6)
        return [ax]

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

    # Determine number of rules with a quiescent state (s_q = 0)
    n_quiescent = (1 - langton_lambda) * n_alpha

    # Determine the indices of rules leading to non-quescent states
    print("Retrieving non-quiescent indices...")
    nonquiescent_indices = np.random.choice(np.arange(n_alpha), int(n_alpha - n_quiescent), replace=False)

    # Fill set of rules
    print("Setting rule states...")
    rule_states = np.zeros(n_alpha, dtype=int)
    rule_states[nonquiescent_indices] = np.random.randint(1, n_states, int(n_alpha - n_quiescent))
    
    # rule_states = np.random.randint(0, n_states, np.power(n_states, neighbourhood_size))

    # return rule_alpha_integers, rule_states
    return rule_states


def construct_ca_quiescent_indices_random(n_dims, n_states, langton_lambda=0.5):
    """Randomly picks the rule indices for the quiescent state (s_q = 0) and returns them.
    This implies a probabilistic selection of the remaining states in the CA algorithm,
    so the rules do not need to be constructed explicitly.
    """

    neighbourhood_size = 3 ** n_dims
    n_alpha = n_states ** neighbourhood_size # possible neighbourhood states

    # Determine number of rules with a quiescent state (s_q = 0)
    n_quiescent = int((1 - langton_lambda) * n_alpha)

    quiescent_indices = unique_random_ints(n_quiescent, n_alpha)

    return quiescent_indices


def run_ca(rule_states, n_steps, n_dims, n_states, space_size, return_steps=False, return_transient=True):
    """Run CA for a set number of steps (n_steps) over a grid of a set size (space_size ^ n_dims)
    using a rule encoded in a numpy array (rule_states) of the rule output states.
    """

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

        # Reconstruct rule indices on lattice by summing neighbour states multiplied by increasing powers of n_states
        shifted_lattices = np.zeros(np.insert(lattice_dims, 0, dim_shifts.shape[0]), dtype=int)
        for j, ds in enumerate(dim_shifts):
            shifted_lattices[j]  = np.roll(step_configs[i-1], ds, np.arange(n_dims)) * (n_states ** j)
        lattice_rule_indices = np.sum(shifted_lattices, axis=0)

        # Determine next configuration of states
        step_configs[i] = rule_states[lattice_rule_indices]

    return step_configs


def run_ca_from_quiescent(quiescent_indices, n_steps, n_dims, n_states, space_size, return_steps=False, return_transient=True):
    """Run CA for a set number of steps (n_steps) over a grid of a set size (space_size ^ n_dims)
    using only the indices of the quiescent states.
    """

    lattice_dims = np.full(n_dims, space_size)
    config_dims = np.insert(lattice_dims, 0, n_steps)
    
    step_configs = np.empty(config_dims, dtype=int)

    # Determine dimension shifts
    dim_shifts = product(range(-1, 2), repeat=n_dims)
    dim_shifts = np.array(list(dim_shifts))
    
    # Initialise random states
    step_configs[0] = np.random.randint(0, n_states, lattice_dims)
    # print(step_configs[0])

    # Create a default random generator
    rand_int_stream = RandomIntStream(1, n_states, 1e6)

    for i in range(1, n_steps):

        # Reconstruct rule indices on lattice by summing neighbour states multiplied by increasing powers of n_states
        shifted_lattices = np.zeros(np.insert(lattice_dims, 0, dim_shifts.shape[0]), dtype=int)
        for j, ds in enumerate(dim_shifts):
            shifted_lattices[j]  = np.roll(step_configs[i-1], ds, np.arange(n_dims)) * (n_states ** j)
        lattice_rule_indices = np.sum(shifted_lattices, axis=0)

        lattice_rule_indices_flat = lattice_rule_indices.flatten()
        rule_states = np.zeros_like(lattice_rule_indices_flat)

        print("Assigning new states...")
        for j, lri in enumerate(lattice_rule_indices_flat):
            if lri in quiescent_indices:
                rule_states[j] = 0
            else:
                rule_states[j] = rand_int_stream.get(lri)
        
        # Determine next configuration of states
        # step_configs[i] = rule_states[lattice_rule_indices]
        step_configs[i] = np.reshape(rule_states, lattice_rule_indices.shape)

    return step_configs


def run_ca_from_lambda(langton_lambda, n_steps, n_dims, n_states, space_size, return_steps=False, return_transient=True):
    """Run CA for a set number of steps (n_steps) over a grid of a set size (space_size ^ n_dims)
    without pre-generated rules, using only the lambda parameter (langton_lambda).
    """

    # CA parameters
    neighbourhood_size = 3 ** n_dims
    n_alpha = n_states ** neighbourhood_size # possible neighbourhood states

    # Determine probability for quiescent state
    quiescent_prob = 1 - langton_lambda

    # Determine max number of rules with a quiescent state (s_q = 0)
    n_quiescent = int(quiescent_prob * n_alpha)

    # Set space-time dimensions
    lattice_dims = np.full(n_dims, space_size)
    config_dims = np.insert(lattice_dims, 0, n_steps)
    
    step_configs = np.empty(config_dims, dtype=int)

    # Determine dimension shifts
    dim_shifts = product(range(-1, 2), repeat=n_dims)
    dim_shifts = np.array(list(dim_shifts))
    
    # Initialise random states
    step_configs[0] = np.random.randint(0, n_states, lattice_dims)
    # print(step_configs[0])

    # Create a default random generator
    rand_int_stream = RandomIntStream(1, n_states, 1e9)

    # Create empty set for storing quiescent indices
    quiescent_indices = set()

    for i in range(1, n_steps):

        # Reconstruct rule indices on lattice by summing neighbour states multiplied by increasing powers of n_states
        shifted_lattices = np.zeros(np.insert(lattice_dims, 0, dim_shifts.shape[0]), dtype=int)
        for j, ds in enumerate(dim_shifts):
            shifted_lattices[j]  = np.roll(step_configs[i-1], ds, np.arange(n_dims))# * (n_states ** j)
        # lattice_rule_indices = np.sum(shifted_lattices, axis=0)
        
        # Add shifted lattices to lists of Python int
        reshaped_shifted_lattices = shifted_lattices.reshape(shifted_lattices.shape[0], -1)
        lattice_rule_indices_flat = []
        for j in range(reshaped_shifted_lattices.shape[1]):
            lattice_rule_indices_flat.append(sum([int(item) * (n_states ** k) for k, item in enumerate(reshaped_shifted_lattices[:,j])]))

        # lattice_rule_indices_flat = lattice_rule_indices.flatten()
        # rule_states = np.zeros_like(lattice_rule_indices_flat)
            
        rule_states = np.zeros(len(lattice_rule_indices_flat))

        print("Assigning new states...")
        for j, lri in enumerate(lattice_rule_indices_flat):
            if lri in quiescent_indices:
                # Quiescent state rule already in collection
                rule_states[j] = 0
            else:
                if np.random.uniform() < quiescent_prob:
                    # New quiescent state rule found
                    quiescent_indices.add(lri)
                    rule_states[j] = 0
                    quiescent_prob = (n_quiescent - len(quiescent_indices)) / n_alpha
                else:
                    # Assign random non-quiescent rule
                    rule_states[j] = rand_int_stream.get(lri)
        
        # Determine next configuration of states
        # step_configs[i] = rule_states[lattice_rule_indices]
        step_configs[i] = np.reshape(rule_states, step_configs.shape[1:])
    # print(quiescent_indices)

    return step_configs


def main():

    # CA parameters
    n_steps = 8
    n_dims = 2
    n_states = 3
    space_size = 8
    langton_lambda = 0.75
    show_evolution = True
    
    # ca_rule = construct_ca_rule_random(n_dims, n_states, langton_lambda)
    # print(ca_rule)

    # ca_steps = run_ca(ca_rule, n_steps, n_dims, n_states, space_size)
    # print(ca_steps[1])

    # quiescent_indices = construct_ca_quiescent_indices_random(n_dims, n_states, langton_lambda)
    # print(quiescent_indices.shape)
    # ca_steps = run_ca_from_quiescent(quiescent_indices, n_steps, n_dims, n_states, space_size)

    ca_steps = run_ca_from_lambda(langton_lambda, n_steps, n_dims, n_states, space_size)

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
            show_state_over_time_3D(ca_steps, n_states)


if __name__ == '__main__':
    main()