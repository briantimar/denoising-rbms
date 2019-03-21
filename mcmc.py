""" tools for doing MCMC sampling """
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def update_ising(v, beta):
    """Given spin state v, propose new state and accept / reject as appropriate
        v: a tensor taking values in -1, 1.
        beta: inverse temp

    Returns: newstate, """
    Nchain = tf.shape(v)[0]
    L = tf.shape(v)[1]

    #(Nchain,) tensor indicating which site is selected for each chain
    indices_to_flip = tf.squeeze(tf.random.categorical(logits=tf.ones((Nchain, L)),
                                        num_samples=1, dtype=tf.int32))
    #indices to the left and right
    left_indices = tf.floormod(indices_to_flip-1, L)
    right_indices = tf.floormod(indices_to_flip+1, L)

    #N,3 tensor of all relevant indices
    indices_all = tf.stack([left_indices, indices_to_flip, right_indices], axis=1)
    #N, 3 tensor indexing the chains
    chain_indices = tf.tile(tf.expand_dims(tf.range(Nchain, dtype=tf.int32), axis=1), [1, 3])
    #N,3,2 tensor
    indices_full = tf.stack([chain_indices, indices_all], axis=2)

    #N, 3 tensor of spin values at selected sites
    relevant_spins = tf.gather_nd(v, indices_full)

    #defined such that p(new)/p(old) = exp(-beta * energy cost)
    #(N,) tensor
    energy_cost = tf.cast(2* relevant_spins[:,1] * (relevant_spins[:,0] +
                                                    relevant_spins[:,2]), tf.float32)
    boltzmann_ratios = tf.exp(-beta * energy_cost)

    paccept = tf.minimum(1, boltzmann_ratios)

    accepted_flips = tf.expand_dims( tf.cast(tfp.distributions.Bernoulli(probs=paccept).sample(),
                            tf.float32), 1)


    #N, L tensor, nonzero at flipped sites
    flip_mask = tf.cast(tf.one_hot(indices_to_flip, depth=L), tf.float32)
    #flipped state
    flippedstate = flip_mask * (-v) + (1-flip_mask) * v
    #updated state
    newstate = accepted_flips * flippedstate + (1- accepted_flips) * v

    return newstate



def build_1d_ising_sampler(L, beta, Nstep, Nchain):
    """ Sampling routine for 1d classical ising hamiltonian
    H = - sum(si * si+1)
    """

    #initial spin configuration
    v = tfp.distributions.Bernoulli(probs=.5 * tf.ones((Nchain, L))).sample()

    #run Nstep
