import numpy as np

def euclidean(a, b):
    return np.linalg.norm(a - b)

def cosine(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def manhattan(a, b):
    return np.sum(np.abs(a - b))

def dot(a, b):
    return -np.dot(a, b)

DISTANCES = {
    "euclidean": euclidean,
    "cosine": cosine,
    "manhattan": manhattan,
    "dot": dot
}


def brute_force_knn(q_vector, vector_dict, k=5, metric="euclidean"):
    dis_fn = DISTANCES[metric]
    scores = []
    for id, vec in vector_dict.items():
        dis = dis_fn(q_vector, vec)
        scores.append((id, dis))

    scores.sort(key=lambda x: x[1])
    return scores[:k]  