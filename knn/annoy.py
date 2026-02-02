import numpy as np
import threading

class AnnoyNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.hyperplane = None 
        self.offset = None
        self.bucket = None 

class AnnoyIndex:
    
    def __init__(self, num_trees=10, max_leaf_size=15):
        self.num_trees = num_trees
        self.max_leaf_size = max_leaf_size
        self.roots = []
        self.vectors = {}
        self.lock = threading.RLock()

    def build(self, vectors):
        with self.lock:
            self.vectors = vectors
            self.roots = []
            ids = list(vectors.keys())
           
            for _ in range(self.num_trees):
                root = self._build_tree(ids)
                self.roots.append(root)

    def _build_tree(self, indices):
   
        if len(indices) <= self.max_leaf_size:
            node = AnnoyNode()
            node.bucket = indices
            return node
        
       
        try:
            idx1, idx2 = np.random.choice(indices, 2, replace=False)
        except ValueError:
          
            node = AnnoyNode()
            node.bucket = indices
            return node

        vec1 = self.vectors[idx1]
        vec2 = self.vectors[idx2]
        
      
        normal = vec1 - vec2
        norm_len = np.linalg.norm(normal)
        if norm_len == 0:
             normal = np.random.randn(vec1.shape[0])
        else:
             normal /= norm_len
             
   
        midpoint = (vec1 + vec2) / 2
        offset = -np.dot(normal, midpoint)
    
        left_idxs = []
        right_idxs = []
        
        for idx in indices:
            dist = np.dot(normal, self.vectors[idx]) + offset
            if dist > 0:
                right_idxs.append(idx)
            else:
                left_idxs.append(idx)
                

        if not left_idxs or not right_idxs:
            node = AnnoyNode()
            node.bucket = indices
            return node
            
        node = AnnoyNode()
        node.hyperplane = normal
        node.offset = offset
        node.left = self._build_tree(left_idxs)
        node.right = self._build_tree(right_idxs)
        return node

    def query(self, query_vec, k=5):
        with self.lock:
            candidates = set()
            
            # Search all trees
            for root in self.roots:
                self._traverse(root, query_vec, candidates)
        
            results = []
            for id in candidates:
                if id in self.vectors:
                    dist = np.linalg.norm(query_vec - self.vectors[id])
                    results.append((id, dist))
            
            results.sort(key=lambda x: x[1])
            return results[:k]

    def _traverse(self, node, vec, candidates):
        if node.bucket is not None:
            candidates.update(node.bucket)
            return
            
        dist = np.dot(node.hyperplane, vec) + node.offset
        

        if dist > 0:
            self._traverse(node.right, vec, candidates)
        else:
            self._traverse(node.left, vec, candidates)
 
    
    def clear(self):
        with self.lock:
            self.roots = []