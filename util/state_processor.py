import numpy as np
import tensorflow as tf


class StateProcessor(object):
    """Process a raw image. Resizes it and covert it to grayscale.

    Args:
        input_shape:        Shape of input image.
        output_shape:       Shape of output image.
    """

    def __init__(self, input_shape, output_shape):
        with tf.variable_scope('state_processor'):
            self.input = tf.placeholder(shape=input_shape, dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input)
            self.output = tf.image.resize_images(self.output, output_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess:       Tensorflow session.
            state:      Input image of shape input_shape.
        """
        return sess.run(self.output, feed_dict={self.input: state})
