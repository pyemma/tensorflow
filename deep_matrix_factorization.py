import tensorflow as tf
import numpy as np

class DeepMatrixFactorization(object):
    """
    A simple implemenation of deep matrix factorization arch

                 neural network
                       |
        [user_embedding, item_embedding]
            /                      \
    user_embeddings               item_embeddings
    """
    def __init__(
        self,
        num_user,
        num_item,
        embedding_dim,
        hidden_dims,
        verbose=False,
    ):
        self.num_user = num_user
        self.num_item = num_item
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.verbose = verbose

        self._init_network()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def _init_network(self):
        tf.reset_default_graph()
        self.user = tf.placeholder(tf.int32, shape=[None], name='user_id')
        self.item = tf.placeholder(tf.int32, shape=[None], name='item_id')
        self.label = tf.placeholder(tf.float32, shape=[None], name='label')

        with tf.variable_scope('embeddings'):
            self.user_embeddings = tf.get_variable('user_embeddings', [self.num_user, self.embedding_dim], initializer=tf.truncated_normal_initializer)
            self.item_embeddings = tf.get_variable('item_embeddings', [self.num_item, self.embedding_dim], initializer=tf.truncated_normal_initializer)

        embeded_users = tf.gather(self.user_embeddings, self.user)
        embeded_items = tf.gather(self.item_embeddings, self.item)
        embeddings = tf.concat([embeded_users, embeded_items], 1)

        perv_dim = self.embedding_dim * 2
        perv_out = embeddings
        for idx, hidden_dim in enumerate(self.hidden_dims):
            with tf.variable_scope('layer%d' % (idx + 1)):
                W = tf.get_variable('W%d' % (idx + 1), [perv_dim, hidden_dim], initializer=tf.truncated_normal_initializer)
                b = tf.get_variable('b%d' % (idx + 1), [hidden_dim], initializer=tf.constant_initializer(0.1))
                relu = tf.nn.relu(tf.matmul(perv_out, W) + b)
                perv_out = relu
                perv_dim = hidden_dim

        with tf.variable_scope('out_layer'):
            W = tf.get_variable('W_out', [perv_dim, 1], initializer=tf.truncated_normal_initializer)
            b = tf.get_variable('b_out', [1], initializer=tf.constant_initializer(0.1))
            self.out = tf.matmul(perv_out, W) + b

        self.loss = tf.reduce_mean(tf.squared_difference(self.label, self.out))
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
