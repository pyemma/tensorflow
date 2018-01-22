import gym
from dqn import DQN


class CartPolePlay(object):

    def __init__(
        self,
        hidden_dims,
        step_to_copy_graph=300,
        step_each_epsiode=500,
    ):
        self.step_to_copy_graph = step_to_copy_graph
        self.step_each_epsiode = step_each_epsiode

        self.dqn = DQN(4, 2, hidden_dims)
        self.env = gym.make('CartPole-v0')

    def train(self, num_train=5000):
        running_score = 0.0
        num_epsiode = 0
        num_step = 0

        for _ in range(num_train):
            state = self.env.reset()

            for t in range(self.step_each_epsiode):
                num_step += 1
                action = self.dqn.action(state)
                next_state, reward, done, _ = self.env.step(action)
                reward = -100 if done else 0.1
                self.dqn.remember(state, action, reward, done, next_state)
                state = next_state

                self.dqn.learn()

                if num_step % self.step_to_copy_graph == 0:
                    self.dqn.copy_graph()

                if done:
                    running_score += t
                    break

            num_epsiode += 1
            self.dqn.decrease_epsilon()
            if num_epsiode % 100 == 0:
                running_score /= 100
                print("Current running score is: %.2f" % running_score)
                if running_score > 195.0:
                    print("HaHa, solved in: %d" % num_epsiode)
                    return True
                running_score = 0.0
        return False


    def play(self, num_epsiode):
        total_score = 0.0
        for _ in range(num_epsiode):
            state = self.env.reset()
            while True:
                action = self.dqn.play(state)
                next_state, reward, done, _ = self.env.step(action)
                if done:
                    break
                total_score += 1
                state = next_state
        return total_score / num_epsiode

    def store(self):
        self.dqn.save('model/dqn/cartpole-v0.ckpt')
