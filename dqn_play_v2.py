from util.fully_connected_nn import FullyConnectedNN
from util.dqn import DQN

import numpy as np
import tensorflow as tf
import gym

q_model = FullyConnectedNN(4, 2, [10, 20, 10], scope='q_model', learning_rate=1e-2)
target_model = FullyConnectedNN(4, 2, [10, 20, 10], scope='target_model', learning_rate=1e-2)

sess = tf.InteractiveSession()
env = gym.make('CartPole-v0')

dqn = DQN(
    sess,
    env,
    q_model,
    target_model,
    [0, 1],
    memory_size=1000,
    epsilon_start=1.0,
    epsilon_end=0.1,
    batch_size=64,
    step_to_copy_graph=100)

sess.run(tf.global_variables_initializer())
dqn.train(epsiode=10000)
