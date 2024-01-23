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

    init = np.zeros((size,size), dtype = int)

    particles_indices = np.random.choice(n_elements, n_particles, replace = False)
    init.flat[particles_indices] = 1

    return init

def move_particles_diffuse(init):
    '''
    Changes the particles in init in a random way. 
    They can move up, down, to the left or to the right.
    Returns the array after one step.
    '''
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    particle_coordinates = np.column_stack(np.where(init == 1))

    for particle in particle_coordinates:
            x, y = particle

            # randomly choose a step
            dx, dy = moves[np.random.choice(len(moves))]
            bound = init.shape[0]

            # update particle position
            new_x, new_y = (x + dx) % bound , (y + dy) % bound # figure wraps around itself

            # if the particle bumps into another particle, it remains where it is
            if new_x in particle_coordinates[:,0] or new_y in particle_coordinates[:,1]:
                 new_x, new_y = x, y
                 
            # update the particle's position and init array
            new_particle_index = new_x * bound + new_y

            init.flat[particle], init.flat[new_particle_index] = 0, 1

    return init

def move_particles_laminar():
    pass

def main():
    pass

if __name__ == "__main__":
    main()