import numpy as np
import os

class RandomHyperplaneHash:
    def __init__(self, dim, num_bits):
        self.dim = dim
        self.num_bits = num_bits
        self.planes = np.random.randn(num_bits, dim)

    def hash(self, vector):
        vector = vector.flatten()
        projections = np.dot(self.planes, vector)
        return tuple(projections > 0)

    def save(self, filepath):
        np.savez(filepath, planes=self.planes, config=np.array([self.dim, self.num_bits]))

    def load(self, filepath):
        if os.path.exists(filepath):
            data = np.load(filepath)
            self.planes = data['planes']
            config = data['config']
            self.dim = config[0]
            self.num_bits = config[1]