from collections import defaultdict
from lsh.hash_functions import RandomHyperplaneHash
import numpy as np
import pickle
import os
import threading

class LSH:
    def __init__(self, dim, num_bits, num_tables, persistence_path="data/lsh_index"):
        self.lock = threading.RLock()
        self.dim = dim
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.persistence_path = persistence_path
        
        self.hash_funcs = []
        self.hash_tables = []

        if self._load():
            print("LSH Index loaded.")
        else:
            print("Initializing...")
            for _ in range(num_tables):
                self.hash_funcs.append(RandomHyperplaneHash(dim, num_bits))
                self.hash_tables.append(defaultdict(list))

    def _get_paths(self):
        if not os.path.exists(os.path.dirname(self.persistence_path)):
             os.makedirs(os.path.dirname(self.persistence_path))
        
        table_path = f"{self.persistence_path}_tables.pkl"
        planes_path_prefix = f"{self.persistence_path}_planes_"
        return table_path, planes_path_prefix

    def save(self):
        with self.lock:
            table_path, planes_path_prefix = self._get_paths()
            with open(table_path, 'wb') as f:
                pickle.dump(self.hash_tables, f)
            for i, hf in enumerate(self.hash_funcs):
                hf.save(f"{planes_path_prefix}{i}.npz")
            print("LSH Index saved.")

    def _load(self):
        with self.lock:
            table_path, planes_path_prefix = self._get_paths()
            if not os.path.exists(table_path): return False
            try:
                with open(table_path, 'rb') as f:
                    self.hash_tables = pickle.load(f)
                self.hash_funcs = []
                for i in range(self.num_tables):
                    hf = RandomHyperplaneHash(self.dim, self.num_bits)
                    hf.load(f"{planes_path_prefix}{i}.npz")
                    self.hash_funcs.append(hf)
                return True
            except Exception as e:
                print(f"Failed to load LSH index: {e}")
                return False

    def add_vector(self, id, vec):
        with self.lock:
            flat_vec = vec.flatten()
            for table, hf in zip(self.hash_tables, self.hash_funcs):
                key = hf.hash(flat_vec)
                table[key].append(id)

    def index(self, vectors):
        with self.lock:
            self.hash_tables = [defaultdict(list) for _ in range(self.num_tables)]
            for id, vec in vectors.items():
                self.add_vector(id, vec)
            self.save()

    def query(self, query_vec, vectors, k=5):
        with self.lock:
            candidates = set()
            flat_query_vec = query_vec.flatten()
            for i, (table, hf) in enumerate(zip(self.hash_tables, self.hash_funcs)):
                key = hf.hash(flat_query_vec)
                candidates.update(table.get(key, []))
            
            scores = []
            for id in candidates:
                if id in vectors:
                    dis = np.linalg.norm(flat_query_vec - vectors[id].flatten())
                    scores.append((id, dis))

            scores.sort(key=lambda x: x[1])
            return scores[:k]

    def clear(self):
        with self.lock:
            self.hash_tables = [defaultdict(list) for _ in range(self.num_tables)]
            self.save()