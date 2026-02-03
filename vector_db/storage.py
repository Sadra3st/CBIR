import numpy as np
import json
import os
import threading

class VectorDB:
    def __init__(self, vector_path, meta_path):
        # RLock allows the same thread to acquire the lock multiple times
        self.lock = threading.RLock()
        
        self.vector_file_path = vector_path
        self.meta_file_path = meta_path
        
        # load vectors
        if os.path.exists(self.vector_file_path):
            try:
                self.vectors = np.load(self.vector_file_path, allow_pickle=True).item()
            except Exception as e:
                print(f"Error loading vectors: {e}")
                self.vectors = {}
        else:
            self.vectors = {}

        # load metadata
        if os.path.exists(self.meta_file_path):
            try:
                with open(self.meta_file_path, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def save(self):
        with self.lock:
            np.save(self.vector_file_path, self.vectors)
            with open(self.meta_file_path, 'w') as f:
                json.dump(self.metadata, f, indent=4)
            print("Database saved to disk.")

    def insert(self, id, vector, meta):
        with self.lock:
            self.vectors[id] = vector
            self.metadata[id] = meta

    def update(self, id, vector=None, meta=None):
        with self.lock:
            if id in self.vectors and vector is not None:
                self.vectors[id] = vector
            
            if id in self.metadata and meta is not None:
                self.metadata[id].update(meta)
            
    def get_vector(self, id):
        with self.lock:
            return self.vectors.get(id)
    
    def get_metadata(self, id):
        with self.lock:
            return self.metadata.get(id)

    def get_all_vectors(self):
        with self.lock:
            return self.vectors.copy()

    def delete(self, id):
        with self.lock:
            deleted = False
            if id in self.vectors:
                del self.vectors[id]
                deleted = True
            if id in self.metadata:
                del self.metadata[id]
                deleted = True
            return deleted

    def clear(self):
        """Wipe all data"""
        with self.lock:
            self.vectors = {}
            self.metadata = {}
            self.save()