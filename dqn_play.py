import gym
from dqn import DQN

env = gym.make('CartPole-v0')

dqn = DQN(4, 2, [10, 20, 10])
running_score = 0.0
num_epsiode = 0
num_steps = 0

for _ in range(1000):
    state = env.reset()

    for t in range(500):
        num_steps += 1
        action = dqn.action(state)
        next_state, reward, done, _ = env.step(action)
        dqn.remember(state, action, reward, done, next_state)
        state = next_state

        dqn.learn()

        if num_steps % 300 == 0:
            dqn.copy_graph()

        if done:
            running_score += t
            break

    num_epsiode += 1
    dqn.decrease_epsilon()

    if num_epsiode % 100 == 0:
        print("Current running score is: %.2f" %(running_score / 100))
        running_score = 0.0
