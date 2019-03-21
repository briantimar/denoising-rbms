
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from mcmc import update_ising
tf.enable_eager_execution()

#system size
L = 50
#number of samples per inv temp
Nsample = int(1e5)
#number of mcmc chains to use
Nchain = Nsample
#number of mcmc steps between samples
steps_between_samples = int(1e4)
samples_per_chain = Nsample//Nchain
Nsteps = samples_per_chain * steps_between_samples

#values of inverse temp to use
beta_vals = np.linspace(.1, 3, 20)

savedir = "data/"

np.save(savedir + "beta", beta_vals)

def get_samples(L, beta, Nchain , samples_per_chain, steps_between_samples,
                burn_in = int(1e4)):
    v = tf.cast(tfp.distributions.Bernoulli(probs=.5 * tf.ones((Nchain, L))).sample(),
                tf.float32)
    v = 2 * v - 1
    newstate = v
    Nsteps = steps_between_samples * samples_per_chain
    samples = np.empty((Nchain, samples_per_chain, L))
    for __ in range(burn_in):
        newstate= update_ising(newstate, beta)
    print("finished burn in")
    for ii in range(Nsteps):
        newstate = update_ising(newstate, beta)
        if ii % steps_between_samples == 0:
            samples[:, ii//steps_between_samples, : ] = newstate.numpy()


    return samples

for jj in range(len(beta_vals)):
    beta = beta_vals[jj]
    print("Sampling from beta=", beta)
    samples = get_samples(L, beta, Nchain, samples_per_chain, steps_between_samples)
    #back to binary
    samples = (samples + 1)/2
    np.save(savedir + "fm_ising_samples_beta_indx_{0}".format(jj), samples.astype(np.int32))
