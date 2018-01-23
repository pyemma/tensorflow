from util.conv_nn import ConvNN
from util.dqn_flappy_bird import DQN
from util.state_processor import StateProcessor


import numpy as np
import tensorflow as tf
import game.wrapped_flappy_bird as game

q_model = ConvNN([80, 80, 3], 2, scope='q_model')
target_model = ConvNN([80, 80, 3], 2, scope='target_model')

sess = tf.InteractiveSession()
game_state = game.GameState()
state_processor = StateProcessor([288, 512, 3], [80, 80])

dqn = DQN(
    sess,
    game_state,
    state_processor,
    q_model,
    target_model,
    [0, 1],
    memory_size=10000,
    epsilon_start=1.0,
    epsilon_decay=0.995,
    batch_size=32,
    step_to_copy_graph=100)

sess.run(tf.global_variables_initializer())
dqn.train(epsiode=100000)
