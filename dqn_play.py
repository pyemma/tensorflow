from dqn import DQN
from cartpole_play import CartPolePlay


MAX_EPSIODE = 5000
STEPS_TO_COPY_GRAPH = 300
STEPS_EACH_EPSIODE = 500
NUM_TRAIN = 10

best_eval_score = 0

for _ in range(5):
    cartpole = CartPolePlay(
        hidden_dims=[10, 20, 10],
        step_to_copy_graph=STEPS_TO_COPY_GRAPH,
        step_each_epsiode=STEPS_EACH_EPSIODE
    )

    solved = cartpole.train(num_train=1000)
    if solved:
        score = cartpole.play(10)

        if score > best_eval_score:
            print("Find new best score %.2f" % score)
            best_eval_score = score
            cartpole.store()
