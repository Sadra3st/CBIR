import numpy as np
from knn.distance import DISTANCES

class BruteForceSearch:
    def __init__(self, metric="euclidean"):
        if metric not in DISTANCES:
            raise ValueError(f"Metric {metric} not supported. Choose from {list(DISTANCES.keys())}")
        self.metric = metric
        self.distance_fn = DISTANCES[metric]

    def search(self, query_vector, db_vectors, k=5):
        """
        Args:
            query_vector: numpy array of shape (dim,)
            db_vectors: dictionary {id: vector}
            k: number of nearest neighbors
        Returns:
            list of tuples (id, distance)
        """
        scores = []

        # Iterate through all vectors in the database
        for id, vec in db_vectors.items():
            # Calculate distance
            dist = self.distance_fn(query_vector, vec)
            scores.append((id, dist))

        # Sort by distance (ascending)
        scores.sort(key=lambda x: x[1])

        # Return top k
        return scores[:k]