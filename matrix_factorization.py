import tensorflow as tf
import numpy as np


class MatrixFactorization(object):
    """
    A simple implemenation of traditional matrix factorization via tensorflow
    """
    def __init__(self, num_user, num_item, latent_vector_dim, verbose=False):
        self.num_user = num_user
        self.num_item = num_item
        self.latent_vector_dim = latent_vector_dim
        self.verbose = verbose

        self._init_graph()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def _init_graph(self):
        tf.reset_default_graph()
        self.user = tf.placeholder(tf.int32, shape=[None], name='user_id')
        self.item = tf.placeholder(tf.int32, shape=[None], name='item_id')
        self.label = tf.placeholder(tf.float32, shape=[None], name='label')

        with tf.variable_scope('latent_vectors'):
            user_latent_vectors = tf.get_variable('user_latent_vector', [self.num_user, self.latent_vector_dim], initializer=tf.truncated_normal_initializer)
            item_latent_vectors = tf.get_variable('item_latent_vector', [self.num_item, self.latent_vector_dim], initializer=tf.truncated_normal_initializer)

        user_vectors = tf.gather(user_latent_vectors, self.user)
        item_vectors = tf.gather(item_latent_vectors, self.item)

        self.out = tf.reduce_sum(tf.multiply(user_vectors, item_vectors), 1)
        self.loss = tf.reduce_mean(tf.squared_difference(self.out, self.label))
        self.train = tf.train.RMSPropOptimizer(1e-2).minimize(self.loss)

    def learn(self, user_ids, item_ids, values, num_training_step, batch_size):
        num_training = user_ids.shape[0]
        for i in range(0, num_training_step):
            batch_idx = np.random.choice(num_training, batch_size)
            self.session.run(self.train, feed_dict={self.user: user_ids[batch_idx].astype(int), self.item: item_ids[batch_idx].astype(int), self.label: values[batch_idx]})

            if self.verbose and i % 100 == 0:
                print(self.session.run(self.loss, feed_dict={self.user: user_ids[batch_idx].astype(int), self.item: item_ids[batch_idx].astype(int), self.label: values[batch_idx]}))

    def test(self, user_ids, item_ids, values):
        return self.session.run(self.loss, feed_dict={self.user: user_ids, self.item: item_ids, self.label: values})
