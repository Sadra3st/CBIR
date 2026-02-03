import numpy as np
import heapq
import threading

class NSWIndex:
    """
    Navigable Small World (NSW) Graph for Approximate Nearest Neighbor Search.
    Nodes are connected to their closest neighbors during insertion (incremental construction).
    Search is performed via Greedy Walk.
    """
    def __init__(self, m=16, ef_construction=50):
        self.m = m 
        self.ef_construction = ef_construction 
        self.graph = {} 
        self.vectors = {} 
        self.enter_point = None
        self.lock = threading.RLock()

    def build(self, vectors):
        """Build graph from scratch"""
        with self.lock:
            self.vectors = vectors
            self.graph = {}
            self.enter_point = None
            
            ids = list(vectors.keys())
            np.random.shuffle(ids)
            
            for doc_id in ids:
                self._insert(doc_id)

    def add_item(self, doc_id, vector):

        with self.lock:
            self.vectors[doc_id] = vector
            self._insert(doc_id)

    def _insert(self, new_id):
        if self.enter_point is None:
            self.graph[new_id] = []
            self.enter_point = new_id
            return

        candidates = self._search_internal(self.vectors[new_id], k=self.m, ef=self.ef_construction)
        
   
        neighbors = [id for id, dist in candidates]
        self.graph[new_id] = neighbors
        
        for neighbor in neighbors:
            if neighbor in self.graph:
                self.graph[neighbor].append(new_id)

                if len(self.graph[neighbor]) > self.m * 2: 
                    self._prune(neighbor)

    def _prune(self, node_id):
        neighbors = self.graph[node_id]
        vec_node = self.vectors[node_id]
        dists = []
        for n in neighbors:
            if n in self.vectors:
                d = np.linalg.norm(vec_node - self.vectors[n])
                dists.append((d, n))
        
        dists.sort()
        self.graph[node_id] = [n for d, n in dists[:self.m]]

    def _search_internal(self, query_vec, k, ef):
        """
        Beam search on graph.
        Returns list of (id, distance) tuples sorted by distance.
        """
        if self.enter_point is None: return []

        start_dist = np.linalg.norm(query_vec - self.vectors[self.enter_point])
        candidates = [(start_dist, self.enter_point)]
        heapq.heapify(candidates)
        
        results = [(-start_dist, self.enter_point)]
        
        visited = {self.enter_point}
        
        while candidates:
    
            curr_dist, curr_id = heapq.heappop(candidates)
            
            furthest_res_dist = -results[0][0]
            
            if curr_dist > furthest_res_dist and len(results) >= ef:
                break
                
            # explore neighbors
            neighbors = self.graph.get(curr_id, [])
            for neighbor in neighbors:
                if neighbor not in visited and neighbor in self.vectors:
                    visited.add(neighbor)
                    
                    dist = np.linalg.norm(query_vec - self.vectors[neighbor])
                    
                    # if neighbor is better than worst result or we haven't filled ef
                    if dist < furthest_res_dist or len(results) < ef:
                        heapq.heappush(candidates, (dist, neighbor))
                        heapq.heappush(results, (-dist, neighbor))
                        
                        # keep result set size <= ef
                        if len(results) > ef:
                            heapq.heappop(results)
                            furthest_res_dist = -results[0][0]

 
        final = [(r[1], -r[0]) for r in results]
        final.sort(key=lambda x: x[1]) # Sort by distance asc
        return final[:k]

    def query(self, query_vec, k=5, ef=50):
        with self.lock:
            if not self.enter_point: return []
            return self._search_internal(query_vec, k, ef)
    
    def clear(self):
        with self.lock:
            self.graph = {}
            self.enter_point = None