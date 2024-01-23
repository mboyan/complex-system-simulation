import numpy as np

def init_particles(size, prop_particles):
    '''
    Gives an array of size x size where prop_particles % is filled with particles.
    1 = particle
    0 = no particle
    '''
    assert 0 <= prop_particles <= 1, 'prop_articles must be a fraction of the particles'

    n_elements = size ** 2
    n_particles = int(prop_particles * n_elements)

    init_array = np.zeros((size,size), dtype = int)

    # initialize the particles randomly
    particles_indices = np.random.choice(n_elements, n_particles, replace = False)
    init_array.flat[particles_indices] = 1

    return init_array

def move_particles_diffuse(init_array):
    '''
    Changes the particles in init_array in a random way. 
    They can move up, down, to the left or to the right.
    Returns the array after one step.
    '''
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    particle_coordinates = np.column_stack(np.where(init_array == 1))
    bound = init_array.shape[0]

    for particle in particle_coordinates:
        x, y = particle
        particle_index = x * bound + y

        # randomly choose a step
        dx, dy = moves[np.random.choice(len(moves))]

        # update particle position
        new_x, new_y = x + dx, (y + dy) % bound  # wrap around horizontally

        if new_x > bound:
            new_x = x

        # if the particle bumps into another particle, it remains where it is
        if (new_x, new_y) not in particle_coordinates.tolist():
            # update the particle's position and init array
            new_particle_index = new_x * bound + new_y
            updated_array = init_array.copy()
            updated_array.flat[particle_index], updated_array.flat[new_particle_index] = 0, 1
        else:
            updated_array = init_array

    return updated_array

def move_particles_laminar():
    pass

def main():
    pass

if __name__ == "__main__":
    main()