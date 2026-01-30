from collections import defaultdict
from lsh.hash_functions import RandomHyperplaneHash
import numpy as np

class LSH:
    def __init__(self, dim, num_bits, num_tables):
        self.hash_funcs = []
        self.hash_tables = []

        for _ in range(num_tables):
            self.hash_funcs.append(RandomHyperplaneHash(dim, num_bits))
            self.hash_tables.append(defaultdict(list))

    def index(self, vectors):
        for id, vec in vectors.items():
            flat_vec = vec.flatten()
            for table, hf in zip(self.hash_tables, self.hash_funcs):
                key = hf.hash(flat_vec)
                table[key].append(id)

    def query(self, query_vec, vectors, k=5):
        candidates = set()

        flat_query_vec = query_vec.flatten()
        
        for table, hf in zip(self.hash_tables, self.hash_funcs):
            key = hf.hash(flat_query_vec)
            candidates.update(table.get(key, []))

        scores = []
        for id in candidates:
            dis = np.linalg.norm(flat_query_vec - vectors[id].flatten())
            scores.append((id, dis))

        scores.sort(key=lambda x: x[1])
        return scores[:k]  