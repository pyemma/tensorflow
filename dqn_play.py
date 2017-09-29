import gym
from dqn import DQN

env = gym.make('CartPole-v0')

MAX_EPSIODE = 5000
STEPS_TO_COPY_GRAPH = 300
STEPS_EACH_EPSIODE = 500
NUM_TRAIN = 10

best_eval_score = 0

for _ in range(2):
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
            reward = -100 if done else 0.1
            dqn.remember(state, action, reward, done, next_state)
            state = next_state

            dqn.learn()

            if num_steps % 300 == 0:
                dqn.copy_graph()

            if done:
                running_score += t
                break;

        num_epsiode += 1
        dqn.decrease_epsilon()
        if num_epsiode % 100 == 0:
            dqn.decrease_epsilon()
            print("Current running score is: %.2f" %(running_score / 100))
            running_score = 0.0

    total_score = 0
    for _ in range(10):
        state = env.reset()
        while True:
            action = dqn.play(state)
            next_state, reward, done, _ = env.step(action)
            if done:
                break
            total_score += 1
            state = next_state
    print("Evalution score: %d" % (total_score))
    if total_score > best_eval_score:
        dqn.save('model/dqn/cartpole-v0.ckpt')
