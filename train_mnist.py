import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from models import LocalNoiseRBM
import time

def train_mnist(rbm, optimizer,
                k=1,ksample=50,
                shuffle=1024,
                batch_size=64,
                epochs=20,
                sample_step=500,
                weight_decay = 0.0,
                noise_condition=False,
                use_self_probs=True,
                persistent=True):

    """Train the provided rbm on the given tf.Dataset"""


    dataset = tfds.load('mnist', with_info=False)['train']


    ## shuffle and batch the dataset
    dataset = dataset.shuffle(shuffle).repeat(epochs).batch(batch_size)
    ## get seeds for persistent chains
    with tf.Session() as sess:
        seed_images = dataset.take(1).make_one_shot_iterator().get_next()['image'].eval()

    seed_images = ((np.reshape(seed_images, (-1, 28**2, 1))/255.0)>.5).astype(np.float32)

    data_feed = tf.cast(dataset.make_one_shot_iterator().get_next()['image'], tf.float32)

    ## reshape to visible layer, and binarize
    data_feed = tf.cast((tf.reshape(data_feed, (-1, 28**2, 1)) / 255.0) >= .5, tf.float32)

    ##placeholder to store persistent states
    persistent_feed = tf.placeholder(dtype=tf.float32, shape=(None, 28**2, 1))
    persistent_state = persistent_feed if persistent else None

    grads_and_vars, new_persistent_state = rbm.nll_gradients(data_feed, k,
                                    weight_decay=weight_decay,
                                    use_self_probs=use_self_probs,
                                    noise_condition=noise_condition,
                                  persistent_state=persistent_state)

    persistent_state_numpy = seed_images

    ## run this op to perform training
    tr_op = optimizer.apply_gradients(grads_and_vars)

    ## for sampling images at the end of training
    sampler_feed = tf.placeholder(dtype=tf.float32, shape=(None, 28**2,1))
    sampler, sample_probs = rbm.build_sampler(sampler_feed, ksample)

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
        final_samples, final_sample_probs = sess.run([sampler, sample_probs], feed_dict = {sampler_feed:seed_images})
        #also, try some with random initialization of the visible state
        final_samples_random_init, final_sample_probs_random_init = sess.run(
                                                    [sampler, sample_probs], feed_dict ={
                                                    sampler_feed: np.random.binomial(1, .5, size=(batch_size, 28**2,1))
                                                        })

        final_samples_all = dict(data_seed=dict(samples=final_samples, probs=final_sample_probs),
                                random_seed=dict(samples=final_samples_random_init,
                                                probs=final_sample_probs_random_init))
    return saved_samples, saved_vars, final_samples_all
