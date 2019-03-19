import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from models import LocalNoiseRBM

def train_mnist(rbm, optimizer,
                k=1,
                shuffle=1024,
                batch_size=64,
                epochs=20,
                weight_decay = 0.0,
                noise_condition=False,
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
    persistent_feed = tf.placeholder(dtype=tf.float32, shape=(batch_size, 28**2, 1))
    persistent_state = persistent_feed if persistent else None

    grads_and_vars, new_persistent_state = rbm.nll_gradients(data_feed, k,
                                    weight_decay=weight_decay,
                                    noise_condition=noise_condition,
                                  persistent_state=persistent_state)

    persistent_state_numpy = seed_images

    ## run this op to perform training
    tr_op = optimizer.apply_gradients(grads_and_vars)

    batch=0
    ## number of batches between samples
    sample_step = 500

    saved_samples = []
    saved_vars = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

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
    return saved_samples, saved_vars



nv = 28**2
nh = 100
rbm = LocalNoiseRBM(nv,nh)
k=10
batch_size=64
epochs=10
persistent=False

lr = 1e-3
weight_decay=1e-4
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
saved_samples, saved_vars = train_mnist(rbm, optimizer, k=k,
                            epochs=epochs,
                            weight_decay=weight_decay,
                            persistent=False,
                            noise_condition=False)
np.save("saved_models/samples_clean_persistent", saved_samples)
np.save("saved_models/weights_clean_persistent", saved_vars)
