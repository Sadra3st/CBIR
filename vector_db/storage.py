import numpy as np
import json
import os

class VectorDB:
    def __init__(self, vector_path, meta_path):
        self.vector_file_path = vector_path
        self.meta_file_path = meta_path
        
        # Load Vectors
        if os.path.exists(self.vector_file_path):
            try:
                # Load numpy file; allow_pickle is needed for dictionaries
                self.vectors = np.load(self.vector_file_path, allow_pickle=True).item()
            except Exception as e:
                print(f"Error loading vectors: {e}")
                self.vectors = {}
        else:
            self.vectors = {}

        # Load Metadata
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
        # Save vectors as .npy
        np.save(self.vector_file_path, self.vectors)
        
        # Save metadata as .json
        with open(self.meta_file_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        print("Database saved to disk.")

    def insert(self, id, vector, meta):
        self.vectors[id] = vector
        self.metadata[id] = meta

    def get_vector(self, id):
        return self.vectors.get(id)
    
    def get_metadata(self, id):
        return self.metadata.get(id)

    def get_all_vectors(self):
        return self.vectors

    def delete(self, id):
        if id in self.vectors:
            del self.vectors[id]
        if id in self.metadata:
            del self.metadata[id]
        self.save()