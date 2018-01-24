import numpy as np
import tensorflow as tf
from collections import namedtuple


class DQN(object):
    """A simple version of Deep Q-Learning Network for flappy bird. TODO: refactor it to make it more general.

    Args:
        sess:               Tensorflow session
        game_state:         GameState of flappy bird
        q_model:            Model for Q-Learning Graph
        target_model:       Model for Q-Target Value Graph
        batch_size:         Batch size for each training step
        epsilon_start:      Maximum value for epsilon
        epsilion_end:       Minimum value for epsilon
        epsilon_decay:      Decay ratio for epsilon
        memory_size:        Size of experience replay memory
        step_to_copy_graph: Step to copy q_model to target_model
        step_each_epsiode:  Step of each epsiode to run
        epsiode:            Step of epsiode to run
    """

    Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])

    def __init__(
        self,
        sess,
        game_state,
        state_processor,
        q_model,
        target_model,
        actions,
        gamma=0.9,
        batch_size=32,
        epsilon_start=0.5,
        epsilon_end=0.01,
        epsilon_decay=0.95,
        memory_size=1000,
        step_to_copy_graph=300,
        step_each_epsiode=200,
    ):
        self.sess = sess
        self.game_state = game_state
        self.state_processor = state_processor
        self.q_model = q_model
        self.target_model = target_model
        self.actions = actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.step_to_copy_graph = step_to_copy_graph
        self.step_each_epsiode = step_each_epsiode

        self.memory = []


    def train(self, epsiode=1000):
        """Train the model

        Args:
            epsiode:        Number of epsiode to train
        """
        num_step = 0
        scores = []
        epsilon = self.epsilon_start

        for ep in range(epsiode):
            frame, reward, done = self.game_state.frame_step([1, 0]) # do nothing at the beginning
            frame = self.state_processor.process(self.sess, frame)
            # state = np.dstack((frame,))
            state = np.dstack((frame, frame, frame))
            score = 0.0
            done = False
            while not done:
                num_step += 1
                action = self._action([state], epsilon)
                actions = np.zeros(len(self.actions))
                actions[action] = 1.0
                next_frame, reward, done = self.game_state.frame_step(actions)
                next_frame = self.state_processor.process(self.sess, next_frame)
                # next_state = np.dstack((next_frame - frame,)) if not done
                if not done:
                    next_state = state
                    next_state[:, :, 0:2] = next_state[:, :, 1:3]
                    next_state[:, :, 2] = next_frame
                else:
                    next_state = state
                    next_state[:, :, 0:2] = next_state[:, :, 1:3]
                    next_state[:, :, 2] = next_state[:, :, 1]
                    reward = -100

                self._remember(state, action, reward, next_state, done)
                state = next_state
                frame = next_frame

                score += 1.0

            self._learn()

            if (ep+1) % self.step_to_copy_graph == 0:
                self._copy_graph()

            if epsilon > self.epsilon_end:
                epsilon *= self.epsilon_decay

            if len(scores) == 100:
                scores.pop(0)

            scores.append(score)

            print("Running score: %.2f" % (sum(scores) / len(scores)))

    # def play(self):
    #     state = self.env.reset()
    #     done = False
    #     step = 0
    #     while not done:
    #         self.env.render()
    #         action = self._action(self._norm(state), 0.0)
    #         state, _, done, _ = self.env.step(action)
    #         step += 1
    #     print("Steps: %d" % step)

    def _action(self, state, epsilon):
        """Use epsilon greedy policy to select action

        Args:
            state:      2d array of shape [1, feat_dim]
            epsilon:    Paramter controlling the exploit/explore effect of epsilon greedy policy
        """
        if np.random.uniform() < epsilon:
            return 1 if np.random.rand() < 0.1 else 0
            # return np.random.randint(0, len(self.actions))
        return np.argmax(self.q_model.predict(self.sess, state))

    def _remember(self, state, action, reward, next_state, done):
        """Remember the experience in memory

            If the size hit the limit, the oldest experience will be forgot
        """
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        self.memory.append(DQN.Experience(state, action, reward, next_state, done))

    def _learn(self):
        """Use Experience Replay and Target Value Network to learn

        """
        if len(self.memory) < self.batch_size:
            return
        sample_idx = np.random.choice(min(len(self.memory), self.memory_size), self.batch_size)
        samples = [self.memory[idx] for idx in sample_idx]

        q_X, target_X, actions, rewards, dones = [], [], [], [], []
        for state, action, reward, next_state, done in samples:
            q_X.append(state)
            target_X.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        q_labels, target_labels = self.q_model.predict(self.sess, np.array(q_X)), self.target_model.predict(self.sess, np.array(target_X))
        q_target = q_labels.copy()
        q_target[np.arange(self.batch_size), np.array(actions)] = np.array(rewards) + self.gamma * np.max(target_labels, axis=1) * (1 - np.array(dones))

        self.q_model.update(self.sess, np.array(q_X), q_target)

    def _copy_graph(self):
        q_params = [t for t in tf.trainable_variables() if t.name.startswith(self.q_model.scope)]
        q_params = sorted(q_params, key=lambda v: v.name)
        t_params = [t for t in tf.trainable_variables() if t.name.startswith(self.target_model.scope)]
        t_params = sorted(t_params, key=lambda v: v.name)

        copy_ops = [tf.assign(t, s) for t, s in zip(t_params, q_params)]
        self.sess.run(copy_ops)
