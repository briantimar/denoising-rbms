import tensorflow as tf
import numpy as np
from models import LocalNoiseRBM
from train_mnist import train_mnist
import json

nv = 28**2
nh = 100
rbm = LocalNoiseRBM(nv,nh)
batch_size=64
epochs=30

persistent=True
noise_condition=False
k=10
ksample=50
sample_step=1000
lr = 1e-3
weight_decay=1e-4
optimizer = tf.train.AdamOptimizer(learning_rate=lr)

config = dict(nv=nv,nh=nh, batch_size=batch_size, epochs=epochs,
                persistent=persistent,k=k,ksample=ksample,
                sample_step=sample_step,lr=lr,weight_decay=weight_decay,
                )

training_samples, saved_vars, final_samples = train_mnist(rbm, optimizer, k=k,
                            ksample=ksample,
                            epochs=epochs,
                            weight_decay=weight_decay,
                            persistent=persistent,
                            noise_condition=noise_condition)

savedir = "saved_models/clean_persistent/"

np.save(savedir+"training_samples",training_samples)
np.save(savedir + "final_samples", final_samples)
np.save(savedir+"variables", saved_vars)
with open(savedir + "config.json", 'w') as f:
    json.dump(config, f)
