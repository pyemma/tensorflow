import numpy as np
import tensorflow as tf


class ConvNN(object):
    """Simple CNN

    Args:
        input_dim:          A list specify the dimension of input image.
        output_dim:         Dimension of output.
        scope:              Scope of all the variables.
        learning_rate:      Learning rate for RMSPropOptimizer
    """

    def __init__(self, input_dim, output_dim, scope='cnn', learning_rate=1e-4):
        self.scope = scope
        with tf.variable_scope(scope):
            self.X_pl = tf.placeholder(tf.uint8, shape=[None]+input_dim, name="X")
            self.y_pl = tf.placeholder(tf.float32, shape=[None, output_dim], name="y")

            X = tf.to_float(self.X_pl) / 255.0

            conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn=tf.nn.relu)
            conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
            conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

            flattened = tf.contrib.layers.flatten(conv3)
            fc1 = tf.contrib.layers.fully_connected(flattened, 512)
            fc2 = tf.contrib.layers.fully_connected(fc1, 64)

            self.predictions = tf.contrib.layers.fully_connected(fc2, output_dim, activation_fn=None)

            self.loss = tf.reduce_mean(tf.squared_difference(self.y_pl, self.predictions))
            self.train = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)

    def predict(self, sess, X):

        return sess.run(self.predictions, feed_dict={self.X_pl: X})

    def update(self, sess, X, y):

        sess.run(self.train, feed_dict={self.X_pl: X, self.y_pl: y})
