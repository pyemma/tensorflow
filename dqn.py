import numpy as np
import tensorflow as tf


class DQN(object):
    """
    A very simple version of Deep Q-Learning Network.

    The Deep Q-Learning Network uses both memory replay and target q network.
    The memory replay is using round replacment to remove the oldest memory.

    feature_dim: number of input feature
    action_dim: number of action
    hidden_dims: each element in the list determines the dimension of the hidden layer
    memory_size: size of memory replay
    batch_size: size of batch samples used for learning
    """

    def __init__(
        self,
        feature_dim,
        action_dim,
        hidden_dims,
        learning_rate = 1e-4,
        gamma = 0.9,
        memory_size = 10000,
        batch_size = 32,
        epsilon = 0.5,
        epsilon_decay = 0.95,
        epsilon_min = 0.01,
    ):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        tf.reset_default_graph()
        self._build_graph()

        self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.eval_out))
        self.train = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        self.memory = np.zeros((self.memory_size, self.feature_dim * 2 + 3))
        self.counter = 0

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())


    def _build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.feature_dim], name='eval_graph_input')
        self.x_ = tf.placeholder(tf.float32, shape=[None, self.feature_dim], name='target_graph_input')
        self.y = tf.placeholder(tf.float32, shape=[None, self.action_dim], name='eval_graph_label')
        self.eval_out = None
        self.target_out = None

        self.W = []
        self.b = []
        self.relu = []

        graph_params = [
            ('eval_graph', self.x, 'eval_graph_params'),
            ('target_graph', self.x_, 'target_graph_params'),
        ]
        for graph_name, graph_input, graph_collections in graph_params:

            graph_W = []
            graph_b = []
            graph_relu = []

            with tf.variable_scope(graph_name):
                collections = [graph_collections, tf.GraphKeys.GLOBAL_VARIABLES]
                prev_out = graph_input
                index = 1
                for i, hidden_dim in enumerate(self.hidden_dims):
                    with tf.variable_scope('layer%d' % index):
                        prev_dim = self.hidden_dims[i-1] if i > 0 else self.feature_dim
                        W = tf.get_variable(
                            'W%d' % index,
                            [prev_dim, hidden_dim],
                            initializer=tf.truncated_normal_initializer,
                            collections=collections,
                        )

                        b = tf.get_variable(
                            'b%d' % index,
                            [hidden_dim],
                            initializer=tf.constant_initializer(0.01),
                            collections=collections,
                        )
                        relu = tf.nn.relu(tf.matmul(prev_out, W) + b)
                        graph_W.append(W)
                        graph_b.append(b)
                        graph_relu.append(relu)
                        prev_out = relu
                        index += 1
                with tf.variable_scope('layerout'):
                    W = tf.get_variable(
                        'Wout',
                        [self.hidden_dims[-1], self.action_dim],
                        initializer=tf.truncated_normal_initializer,
                        collections=collections,
                    )
                    b = tf.get_variable(
                        'bout',
                        [self.action_dim],
                        initializer=tf.truncated_normal_initializer,
                        collections=collections,
                    )
                    out = tf.matmul(prev_out, W) + b
                    graph_W.append(W)
                    graph_b.append(b)
                    graph_relu.append(out)
                    if graph_name == 'eval_graph':
                        self.eval_out = out
                    else:
                        self.target_out = out

                self.W.append(graph_W)
                self.b.append(graph_b)
                self.relu.append(graph_relu)

    def copy_graph(self):
        target_graph_params = tf.get_collection('target_graph_params')
        eval_graph_params = tf.get_collection('eval_graph_params')
        self.sess.run([tf.assign(t, s) for t, s in zip(target_graph_params, eval_graph_params)])

    def decrease_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, done, next_state):
        self.memory[(self.counter % self.memory_size), :] = np.hstack((state, action, reward, done, next_state))
        self.counter += 1

    def action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, self.action_dim-1)
        return np.argmax(self.eval_out.eval(feed_dict={self.x: np.reshape(state, [1, 4])}))

    def learn(self):
        samples = self.memory[np.random.choice(min(self.counter, self.memory_size), self.batch_size)]
        eval_samples = samples[:, : self.feature_dim]
        target_samples = samples[:, -self.feature_dim :]
        eval_labels, target_labels = self.sess.run(
            [self.eval_out, self.target_out],
            feed_dict={self.x: eval_samples, self.x_: target_samples},
        )
        eval_act = samples[:, self.feature_dim].astype(int)
        reward = samples[:, self.feature_dim+1]
        done = samples[:, self.feature_dim+2]

        q_target = eval_labels.copy()
        q_target[np.arange(self.batch_size), eval_act] = reward + self.gamma * np.max(target_labels, axis=1) * (1 - done)
        self.sess.run(self.train, feed_dict={self.x: eval_samples, self.y: q_target})
