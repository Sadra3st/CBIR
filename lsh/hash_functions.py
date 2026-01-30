import numpy as np

class RandomHyperplaneHash:
    def __init__(self, dim, num_bits):
        self.planes = np.random.randn(num_bits, dim)

    def hash(self, vector):
        vector = vector.flatten()
        projections = np.dot(self.planes, vector)
        return tuple(projections > 0)