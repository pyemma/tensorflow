import numpy as np
import tensorflow as tf


class FullyConnectedNN(object):
    """Simple Fully Connected Neural Network

    Args:
        input_dim:      dimension of input training sample
        output_dim:     dimension of output
        hidden_dims:    a list of dimension of hidden layer
        scope:          scope of all the variables live in
        trainable:      whether the variables are trainable or not
        learning_rate:  learning rate for RMSPropOptimizer
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        scope='fully_connected_nn',
        trainable=True,
        learning_rate=1e-4,
    ):
        self.scope = scope
        with tf.variable_scope(scope):
            self.X_pl = tf.placeholder(tf.float32, shape=[None, input_dim], name="X")
            self.y_pl = tf.placeholder(tf.float32, shape=[None, output_dim], name="y")

            perv = self.X_pl
            for hidden_dim in hidden_dims:
                perv = tf.contrib.layers.fully_connected(perv, hidden_dim, trainable=trainable)

            self.predictions = tf.contrib.layers.fully_connected(perv, output_dim, activation_fn=None)

            self.loss = tf.reduce_mean(tf.squared_difference(self.y_pl, self.predictions))
            self.train = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)

    def predict(self, sess, X):
        """Predict output

        Args:
            sess:       Tensorflow session
            X:          A batch of samples of shape [batch_size, input_dim]
        """

        return sess.run(self.predictions, feed_dict={self.X_pl: X})

    def update(self, sess, X, y):
        """Update the Nerual Network

        Args:
            sess:       Tensorflow session
            X:          A batch of samples of shape [batch_size, input_dim]
            y:          A batch of labels of shape [batch_size, output_dim]
        """
        sess.run(self.train, feed_dict={self.X_pl: X, self.y_pl: y})
