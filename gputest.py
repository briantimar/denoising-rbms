import tensorflow as tf
print(tf.__version__)

a = tf.ones(4)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(a))
