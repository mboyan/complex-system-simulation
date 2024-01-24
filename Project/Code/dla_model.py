import numpy as np
from itertools import product


# ===== Particle / lattice initialization =====

def init_lattice(lattice_size, seed_coords):
    """
    Creates a square lattice with initial seeds placed on specific sites.
    inputs:
        lattice_size (int) - the size of the lattice along one dimension
        seed_coords (np.ndarray) - an array of lattice site coordinates for the placement of initial seeds
    """

    # Infer dimensions from number of seed coordinates
    lattice_dims = seed_coords.shape[1]
    assert lattice_dims > 1

    lattice = np.zeros(np.repeat(lattice_size, lattice_dims))

    lattice[tuple(seed_coords.T)] = 1

    return lattice


def init_particles(lattice, prop_particles):
    """
    Creates an array of n-dimensional particles where the number of particles
    is determined by a proportion from an input lattice size.
    inputs:
        lattice (np.ndarray) - an array of lattice sites containing 1's where there are seeds and 0's otherwise
        prop_particles (float) - a number between 0 and 1 determining the percentage / density of particles on the lattice
    outputs:
        init_particles (np.ndarray) - an array of particle coordinates
    """
    assert 0 <= prop_particles <= 1, 'prop_articles must be a fraction of the particles'

    # Find empty locations in the lattice
    empty_locs = np.argwhere(lattice == 0)

    # Determine number of particles=
    n_particles = int(empty_locs.shape[0] * prop_particles)

    # Initialize particles randomly wherever there are no seeds
    init_coords = empty_locs[np.random.choice(empty_locs.shape[0], size=n_particles, replace=False)]

    return init_coords


def regen_particles(lattice, n_particles):
    """
    Randomly regenerate a specific number of particles.
    inputs:
        lattice (np.ndarray) - an array of lattice sites containing 1's where there are seeds and 0's otherwise
        n_particles (int) - the number of particles to regenerate
    """

    # Find empty locations in the lattice
    empty_locs = np.argwhere(lattice == 0)

    assert n_particles <= empty_locs.shape[0], 'too many particles to regenerate'

    # Initialize particles randomly wherever there are no seeds
    regen_coords = empty_locs[np.random.choice(empty_locs.shape[0], size=n_particles, replace=False)]

    return regen_coords


# ===== Particle movement functions =====

def move_particles_diffuse(particles_in, lattice, periodic=(True, True), moore=False):
    """
    Petrurbs the particles in init_array using a Random Walk.
    inputs:
        particles_in (numpy.ndarray) - an array of coordinates
        lattice (np.ndarray) - an array of lattice sites containing 1's where there are seeds and 0's otherwise
        periodic (tuple of bool) - defines whether the lattice is periodic in each dimension, number of elements must correspond to dimensions
        moore (bool) - determine whether the particles move in a Moore neighbourhood
            or otherwise in a von Neumann neighbourhood; defaults to False
    outputs:
        the particle array after one step (numpy.ndarray)
    """
    assert len(periodic) == particles_in.shape[1], 'dimension mismatch with periodicity tuple'

    lattice_dims = np.ndim(lattice)
    lattice_size = lattice.shape[0]

    # Define Moore neighbourhood
    moves = list(product([0, 1, -1], repeat=lattice_dims))
    
    # Reduce to von Neumann neighbourhood
    if not moore:
        moves = [m for m in moves if abs(sum(m)) == 1]

    # Perturb particles (only if they are not on an occupied site)
    moves = np.array(moves)
    perturbations = moves[np.random.randint(len(moves), size=particles_in.shape[0])]
    mask = lattice[tuple(particles_in.T)] == 0
    particles_out = np.array(particles_in)
    particles_out[mask] += perturbations[mask]

    # Wrap around or regenerate
    if not np.all(np.array(periodic)):
        particles_regen = regen_particles(lattice, particles_in.shape[0])
    
    for i, p in enumerate(periodic):
        if p:
            particles_out[:, i] = np.mod(particles_out[:, i], lattice_size)
        else:
            particles_out[:, i] = np.where(np.any((particles_out < 0) | (particles_out >= lattice_size), axis=1), particles_regen[:, i], particles_out[:, i])

    return particles_out


def move_particles_laminar():
    pass


# ===== Aggregation function =====
def aggregate_particles(particles, lattice, prop_particles=None, periodic=(True, True), moore=False):
    """
    Check if particles are neighbouring seeds on the lattice.
    If they are, place new seeds.
    inputs:
        particles (np.ndarray) - an array of particle coordinates
        lattice (np.ndarray) - an array of lattice sites containing 1's where there are seeds and 0's otherwise
        prop_particles (float) - a number between 0 and 1 determining the percentage / density of particles on the lattice;
            if None, the particles will not be regenerated to compensate the proportion
        periodic (tuple of bool) - defines whether the lattice is periodic in each dimension, number of elements must correspond to dimensions
        moore (bool) - determine whether the neighbourhood is Moore or otherwise von Neumann; defaults to False
    """

    lattice_dims = np.ndim(lattice)
    assert lattice_dims == particles.shape[1], 'dimension mismatch between lattice and particles'
    assert len(periodic) == lattice_dims, 'dimension mismatch between periodicity labels and lattice'
    assert len(periodic) == particles.shape[1], 'dimension mismatch between periodicity labels and particles'

    # Define particle neighbourhoods (Moore)
    nbrs = list(product([0, 1, -1], repeat=lattice_dims))

    # Reduce to von Neumann neighbourhood
    if not moore:
        nbrs = [n for n in nbrs if abs(sum(n)) == 1]

    # Shift lattice by neighbourhoods
    nbrs = np.array(nbrs)
    shifted_lattices = np.array([np.roll(lattice, shift, tuple(range(lattice_dims))) for shift in nbrs])
    summed_nbrs_lattice = np.sum(shifted_lattices, axis=0)

    # Zero edge rows of shifted lattices if not periodic
    if not np.all(np.array(periodic)):
        pass
    
    # Check if particles are neighbouring seeds
    new_seed_indices = np.argwhere(summed_nbrs_lattice[tuple(particles.T)] > 0)

    # Update lattice (add seeds)
    lattice[tuple(particles[new_seed_indices].T)] = 1

    # Compensate particle density
    if prop_particles is not None:
        # Recalculate lattice vacancy
        lattice_vacancy = np.argwhere(lattice == 0)
        n_particles_potential = int(lattice_vacancy.shape[0] * prop_particles)

        # Regenerate as many particles as are needed to maintain the proportion
        mask = lattice[tuple(particles.T)] == 0
        active_particle_indices = np.flatnonzero(mask)
        inactive_particle_indices = np.flatnonzero(~mask)
        n_particles_deficit = n_particles_potential - active_particle_indices.shape[0]
        if n_particles_deficit > 0:
            particles_regen = regen_particles(lattice, n_particles_deficit)
            particles[inactive_particle_indices[:n_particles_deficit]] = particles_regen

    return lattice, particles


# ===== Utility functions =====

def particles_to_lattice(particles, lattice_size):
    """
    Projects the particle coordinates on a lattice,
    returning a grid with 1's where there is a particle and 0's otherwise
    """
    assert np.max(particles) < lattice_size, 'mismatch between particle coordinates and lattice size'

    # Infer dimensions from number of particle coordinates
    lattice_dims = particles.shape[1]

    particle_lattice = np.zeros(np.repeat(lattice_size, lattice_dims))

    particle_lattice[tuple(particles.T)] = 1

    return particle_lattice

