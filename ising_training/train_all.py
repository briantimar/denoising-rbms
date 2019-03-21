"""to run:
change dir to denoising-rbms.
then:
> gpurun python -i ising_training/train_all.py

:/
"""
import sys
sys.path.append('/tmp')
import tensorflow as tf
import numpy as np
from models import LocalNoiseRBM
from ising_training.train_ising import train_ising
import json

nv = 50
nh = 25
rbm = LocalNoiseRBM(nv,nh)
batch_size=64
epochs=30

persistent=False
noise_condition=False
use_self_probs=False
k=10
Nsample=32
ksample=50
sample_step=1000
lr = 1e-2
weight_decay=1e-4
optimizer = tf.train.AdamOptimizer(learning_rate=lr)

beta = np.load("ising_training/data/beta.npy")
Nbeta = len(beta)

config = dict(nv=nv,nh=nh, batch_size=batch_size, epochs=epochs,
                persistent=persistent,k=k,ksample=ksample,
                sample_step=sample_step,lr=lr,weight_decay=weight_decay,
                noise_condition=noise_condition,use_self_probs=use_self_probs
                )

savedir = "ising_training/saved_models/"
with open(savedir + "config.json", 'w') as f:
    json.dump(config, f)

for ii in range(Nbeta):
    print("Training on beta=", beta[ii])
    fpath="ising_training/data/fm_ising_samples_beta_indx_{0}.npy".format(ii)
    training_samples, saved_vars, final_samples, sample_seeds, training_seeds,randseed = train_ising(
                                rbm, optimizer, fpath,
                                k=k,
                                batch_size=batch_size,
                                ksample=ksample,
                                Nsample=Nsample,
                                epochs=epochs,
                                weight_decay=weight_decay,
                                persistent=persistent,
                                use_self_probs=use_self_probs,
                                noise_condition=noise_condition)

    def write(x, fname):
        np.save(savedir + fname + "_beta_indx_{0}".format(ii), x)

    write(training_samples, "training_samples")
    write(final_samples, "final_samples")
    write(training_seeds, "training_seeds")
    write(sample_seeds, "sample_seeds")
    write(randseed, "randseed")
    write(saved_vars, "variables")
