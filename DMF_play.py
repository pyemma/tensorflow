import numpy as np
import pandas as pd
import math
from deep_matrix_factorization import DeepMatrixFactorization
from matrix_factorization import MatrixFactorization

def proprocess():
    data = pd.read_csv('the-movies-dataset/ratings_small.csv')
    user_ids = data['userId']
    movie_ids = data['movieId']
    ratings = data['rating']

    movie_id_reindex = {}
    for movie_id in movie_ids:
        if movie_id not in movie_id_reindex:
            idx = len(movie_id_reindex)
            movie_id_reindex[movie_id] = idx

    data_reindex = []
    for user_id, movie_id, rating in zip(user_ids, movie_ids, ratings):
        data_reindex.append(np.array([user_id, movie_id_reindex[movie_id], rating]))
    return np.array(data_reindex)

NUM_MOVIE = 10000
NUM_USER = 700
NUM_TEST = 10
EMBEDDING_DIM = 32
HIDDEN_DIM = [128, 64]

data = proprocess()
num_data = data.shape[0]
num_training = math.ceil(num_data * 0.8)
idx = np.arange(num_data)
np.random.shuffle(idx)
training = data[idx[:num_training], :]
testing = data[idx[num_training:], :]

best_score = math.inf
for _ in range(NUM_TEST):
    dmf = DeepMatrixFactorization(NUM_USER, NUM_MOVIE, EMBEDDING_DIM, HIDDEN_DIM)
    dmf.learn(training[:, 0], training[:, 1], training[:, 2], 2000, 500)

    score = dmf.test(testing[:, 0], testing[:, 1], testing[:, 2])
    if best_score > score:
        best_score = score

print("Best score for deep matrix factorization is %f" % best_score)

best_score = math.inf
for _ in range(NUM_TEST):
    mf = MatrixFactorization(NUM_USER, NUM_MOVIE, EMBEDDING_DIM)
    mf.learn(training[:, 0], training[:, 1], training[:, 2], 2000, 500)

    score = mf.test(testing[:, 0], testing[:, 1], testing[:, 2])
    if best_score > score:
        best_score = score
print("Best score for matrix factorization is %f" % best_score)
