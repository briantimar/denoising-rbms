import sys
sys.path.append('..')
import tensorflow as tf

import numpy as np
from models import LocalNoiseRBM
import time

def train_ising(rbm, optimizer,
                fpath,
                k=1,ksample=50,
                shuffle=1024,
                batch_size=64,
                Nsample=10,
                epochs=20,
                sample_step=500,
                weight_decay = 0.0,
                noise_condition=False,
                use_self_probs=True,
                persistent=True):

    """Train the provided rbm on the given dataset"""

    dataset_numpy = np.load(fpath)
    #system size
    L=dataset_numpy.shape[-1]
    assert L == rbm.num_visible
    dataset_numpy = np.reshape(dataset_numpy, (-1, L,1)).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(dataset_numpy)

    ## shuffle and batch the dataset
    dataset = dataset.shuffle(shuffle).repeat(epochs).batch(batch_size)

    ## get seeds for gibbs chains
    with tf.Session() as sess:
        seedset = dataset.take(2)
        seedfeed = seedset.make_one_shot_iterator().get_next()
        training_seeds = seedfeed.eval()
        sample_seeds = seedfeed.eval()[:Nsample, ...]


    data_feed = tf.reshape(tf.cast(dataset.make_one_shot_iterator().get_next(), tf.float32),
                            (-1, L, 1))

    ##placeholder to store persistent states, if desired
    persistent_feed = tf.placeholder(dtype=tf.float32, shape=(None, L, 1))
    persistent_state = persistent_feed if persistent else None

    grads_and_vars, new_persistent_state = rbm.nll_gradients(data_feed, k,
                                    weight_decay=weight_decay,
                                    use_self_probs=use_self_probs,
                                    noise_condition=noise_condition,
                                  persistent_state=persistent_state)

    persistent_state_numpy = training_seeds

    ## run this op to perform training
    tr_op = optimizer.apply_gradients(grads_and_vars)

    ## for sampling images at the end of training
    sample_steps = list(range(ksample))
    sampler_feed = tf.placeholder(dtype=tf.float32, shape=(None, L,1))
    sampler, sample_probs, intermediate_samples = rbm.build_sampler(sampler_feed, ksample,
                                            sample_steps=sample_steps)

    batch=0

    saved_samples = []
    saved_vars = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        #train the model, record weights and internal samples periodically
        while True:

            try:

                if batch % sample_step == 0:
                    print("sample %d" % (batch// sample_step))
                    saved_samples.append(persistent_state_numpy)
                    saved_vars.append(sess.run(rbm.variables))

                __, persistent_state_numpy = sess.run([tr_op, new_persistent_state],
                                                   feed_dict = {persistent_feed:
                                                                persistent_state_numpy}
                                                    )
                batch +=1

            except tf.errors.OutOfRangeError:
                break


        #record samples with higher 'k' values from trained model
        samples_dataseed = sess.run(intermediate_samples, feed_dict = {sampler_feed:sample_seeds})
        #also, try some with random initialization of the visible state
        randseed=np.random.binomial(1, .5, size=(Nsample, L,1))
        samples_randseed = sess.run( intermediate_samples, feed_dict ={
                                                    sampler_feed:randseed
                                                        })

        final_samples = dict(dataseed=samples_dataseed, randseed=samples_randseed)
    return saved_samples, saved_vars, final_samples, sample_seeds, training_seeds, randseed
