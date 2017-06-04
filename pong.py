import numpy as np
import tensorflow as tf
import gym

# Hyperparameters
D = 80 * 80 # Dimision of input image
H = 200 # Number of hidden layer neurons
batch_size = 10 # Every how many episodes to do a param update
learning_rate = 1e-4
gamma = 0.99 # Discount factor for reward
decay_rate = 0.99 # Decay factor for RMSProp leaky sum of grad^2
render = True
save_path = 'models/pong.ckpt'
MAX_EPISODE_NUMBER = 200

def discount_rewards(r):
    """Take 1D float array of rewards and compute discounted reward"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def prepro(I):
    """Prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector"""
    I = I[35:195] # Crop
    I = I[::2, ::2, 0] # Downsample by factor of 2
    I[I == 144] = 0 # Erase background
    I[I == 109] = 0 # Erase background
    I[I != 0] = 1 # everything else just set to 1
    return I.astype(np.float).ravel()

# Model initialization
W1 = tf.Variable(tf.truncated_normal([D, H], mean=0, stddev=1./np.sqrt(D), dtype=tf.float32))
W2 = tf.Variable(tf.truncated_normal([H, 1], mean=0, stddev=1./np.sqrt(H), dtype=tf.float32))
x = tf.placeholder(dtype=tf.float32, shape=[None, D])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
discounted_rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1])

fc1 = tf.matmul(x, W1)
relu = tf.nn.relu(fc1)
fc2 = tf.matmul(relu, W2)
# Calculate probability which is used for sample action
sig = tf.nn.sigmoid(fc2)
# Train the policy network according the reward we get in final
loss = tf.nn.l2_loss(y - sig)
optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay_rate)
grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=discounted_rewards)
train = optimizer.apply_gradients(grads)

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
xs, ys, drs = [], [], []
reward_sum = 0
running_reward = 0
episode_number = 0

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

saver = tf.train.Saver(tf.all_variables())
load_was_success = True
try:
    save_dir = '/'.join(save_path.split('/')[:-1])
    ckpt = tf.train.get_checkpoint_state(save_dir)
    load_path = ckpt.model_checkpoint_path
    saver.restore(sess, load_path)
except Exception:
    print("No saved model to load, starting new session")
    load_was_success = False
else:
    print("Load model: {}".format(load_path))
    saver = tf.train.Saver(tf.all_variables())
    episode_number = int(load_path.split('-')[-1])


while True:
    if episode_number >= MAX_EPISODE_NUMBER:
        break

    if render:
        env.render()

    # process the observation, set input to network to be difference image
    cur_x = prepro(observation)
    tf_x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    tf_x = np.reshape(tf_x, (1, -1))
    prev_x = cur_x

    feed_dict={x: tf_x}
    aprob = sess.run(sig, feed_dict=feed_dict)
    action = 2 if np.random.uniform() < aprob else 3

    # record various intermediates
    xs.append(tf_x)
    tf_y = 1 if action == 2 else 0
    ys.append(tf_y)

    # step the enviornment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)

    if done:
        episode_number += 1
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

        # stack everything together
        epx = np.vstack(xs)
        epy = np.vstack(ys)
        epr = np.vstack(drs)
        xs, ys, drs = [], [], []

        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        feed_dict = {x: epx, y: epy, discounted_rewards: discounted_epr}
        sess.run(train, feed_dict=feed_dict)

        observation  = env.reset()
        prev_x = None

        if episode_number % 10 == 0:
            print('ep {}: reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward))

        if episode_number % 50 == 0:
            saver.save(sess, save_path, global_step=episode_number)
            print("SAVE MODEL #{}".format(episode_number))
