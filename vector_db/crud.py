import os
import uuid
import numpy as np
import base64
import io
import threading
import time
import random
from PIL import Image
from vector_db.storage import VectorDB
from embedding.resnet import load_resnet18, image_to_embedding
from lsh.lsh import LSH
from knn.brute_force import BruteForceSearch
from knn.nsw import NSWIndex
from knn.annoy import AnnoyIndex

class ImageRetriever:
    def __init__(self, vector_db_path="data/vectors.npy", meta_db_path="data/metadata.json"):
        self.lock = threading.RLock()
        
        self.db = VectorDB(vector_db_path, meta_db_path)
        self.model = load_resnet18()
        
        # Exact Search
        self.bf_search = BruteForceSearch(metric="euclidean")
        
        # LSH (Hash-Based)
        self.lsh = LSH(dim=512, num_bits=6, num_tables=4, persistence_path="data/lsh_index")
        
        # NSW (Graph-Based - Bonus)
        self.nsw = NSWIndex(m=16, ef_construction=100)
        
        # Annoy (Tree-Based - Bonus)
        self.annoy = AnnoyIndex(num_trees=15)
        
        self.indexes_ready = False
        self.indexing_status = "Initializing..."

        # Start Index Building in Background
        threading.Thread(target=self._async_startup, daemon=True).start()

    def _async_startup(self):
        with self.lock:
            # Load LSH 
            if len(self.db.get_all_vectors()) > 0:
                self.indexing_status = "Loading LSH..."
                if not self.lsh.hash_tables[0]:
                    self.refresh_indices(full_rebuild=False)
                else:
        
                    self.rebuild_memory_indexes()
            
            self.indexes_ready = True
            self.indexing_status = "Ready"
            print("Background indexing complete.")

    def refresh_indices(self, full_rebuild=True):
        """Full rebuild of all indexes"""
        with self.lock:
            self.indexing_status = "Indexing..."
            print("Rebuilding all indexes...")
            all_vectors = self.db.get_all_vectors()
            
            if all_vectors:
                # LSH
                if full_rebuild:
                    self.lsh.index(all_vectors)
                self.nsw.build(all_vectors)
                self.annoy.build(all_vectors)
                
            self.indexes_ready = True
            self.indexing_status = "Ready"
            print(f"Indexes built with {len(all_vectors)} vectors.")

    def rebuild_memory_indexes(self):
        with self.lock:
            self.indexing_status = "Building Graphs..."
            print("Building memory-based indexes (NSW, Annoy)...")
            all_vectors = self.db.get_all_vectors()
            if all_vectors:
                self.nsw.build(all_vectors)
                self.annoy.build(all_vectors)

    def add_image(self, image_path, category="unknown"):
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return False

        try:
            embedding = image_to_embedding(self.model, image_path)
            
            thumbnail_b64 = None
            try:
                with Image.open(image_path) as img:
                    img.thumbnail((200, 200))
                    if img.mode != 'RGB': img = img.convert('RGB')
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=70)
                    thumbnail_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            except: pass

            img_id = str(uuid.uuid4())
            meta = {
                "path": image_path,
                "category": category,
                "filename": os.path.basename(image_path),
                "thumbnail": thumbnail_b64
            }

            with self.lock:
                self.db.insert(img_id, embedding, meta)
                self.db.save()
                
                self.lsh.add_vector(img_id, embedding)
                self.lsh.save()
                
                # Update Bonus Indexes
                self.nsw.add_item(img_id, embedding)
                # Annoy doesn't support incremental easily, we skip or rebuild
        
            return img_id
        except Exception as e:
            print(f"Failed to add image: {e}")
            return None
            
    def import_batch(self, embeddings, paths, categories, thumbnails=None):
        count = len(embeddings)
        if thumbnails is None: thumbnails = [None] * count

        print(f"Starting batch import of {count} items...")
        
        with self.lock:
            for i, (vec, path, cat, thumb) in enumerate(zip(embeddings, paths, categories, thumbnails)):
                img_id = str(uuid.uuid4())
                meta = { "path": path, "category": cat, "filename": os.path.basename(path), "thumbnail": thumb }
                self.db.insert(img_id, vec, meta)
                if (i + 1) % 1000 == 0: print(f"Imported {i + 1}...")

            print("Saving DB...")
            self.db.save()
            print("Building Indexes...")
            # We call refresh synchronously here because import is a blocking op anyway
            self.refresh_indices()
            
        print("Batch import complete.")
        return True

    def get_image_details(self, img_id):
        with self.lock:
            return {"vector": self.db.get_vector(img_id), "metadata": self.db.get_metadata(img_id)}

    def delete_image(self, img_id):
        with self.lock:
            if self.db.delete(img_id):
                self.db.save()
                self.refresh_indices()
                return True
            return False

    def reset_database(self):
        with self.lock:
            self.db.clear()
            self.lsh.clear()
            self.nsw.clear()
            self.annoy.clear()
            print("Database reset successfully.")
            return True

    def update_image_metadata(self, img_id, new_category=None):
        with self.lock:
            updates = {}
            if new_category: updates["category"] = new_category
            self.db.update(img_id, meta=updates)
            self.db.save()
            return True

    def search(self, query_image_path, k=5, method="brute_force"):
        query_vec = image_to_embedding(self.model, query_image_path)
        return self.search_by_vector(query_vec, k, method)

    def search_by_vector(self, query_vec, k=5, method="brute_force"):
        results = []
        
        # Guard against using not-ready indexes
        if method in ["nsw", "annoy"] and not self.indexes_ready:
            print("Warning: Advanced indexes not ready. Falling back to Brute Force.")
            method = "brute_force"

        with self.lock:
            vectors = self.db.get_all_vectors()
            
            if method == "lsh":
                results = self.lsh.query(query_vec, vectors, k=k)
            elif method == "nsw":
                results = self.nsw.query(query_vec, k=k)
            elif method == "annoy":
                results = self.annoy.query(query_vec, k=k)
            else: # brute_force
                results = self.bf_search.search(query_vec, vectors, k=k)

            enriched_results = []
            for img_id, score in results:
                meta = self.db.get_metadata(img_id)
                if meta:
                    enriched_results.append({
                        "id": img_id,
                        "score": score,
                        "path": meta.get("path"),
                        "category": meta.get("category"),
                        "thumbnail": meta.get("thumbnail")
                    })
            return enriched_results
    
    def get_all_embeddings_for_viz(self):
        with self.lock:
            ids = []
            vecs = []
            labels = []
            for id, vec in self.db.get_all_vectors().items():
                ids.append(id)
                vecs.append(vec)
                meta = self.db.get_metadata(id)
                labels.append(meta.get("category", "unknown"))
            return ids, np.array(vecs), labels

    def benchmark_algorithms(self, num_queries=50, k=10):
        if not self.indexes_ready:
            return "Indexes are still building. Please wait..."

        with self.lock:
            vectors = self.db.get_all_vectors()
            if len(vectors) < num_queries:
                return "Not enough data to benchmark."
            
            ids = list(vectors.keys())
            query_ids = random.sample(ids, num_queries)
            
            methods = ["lsh", "nsw", "annoy"]
            results = {m: {"time": 0.0, "hits": 0} for m in methods}
            bf_time = 0.0
            
            print(f"Benchmarking {num_queries} queries...")
            
            for q_id in query_ids:
                query_vec = vectors[q_id]
                
                # 1. Ground Truth (Brute Force)
                t0 = time.time()
                bf_res = self.bf_search.search(query_vec, vectors, k=k)
                bf_time += (time.time() - t0)
                ground_truth_ids = set([r[0] for r in bf_res])
                
                # 2. Test Approx Methods
                for m in methods:
                    t0 = time.time()
                    if m == "lsh": approx_res = self.lsh.query(query_vec, vectors, k=k)
                    elif m == "nsw": approx_res = self.nsw.query(query_vec, k=k)
                    elif m == "annoy": approx_res = self.annoy.query(query_vec, k=k)
                    
                    results[m]["time"] += (time.time() - t0)
                    
                    found_ids = set([r[0] for r in approx_res])
                    hits = len(ground_truth_ids.intersection(found_ids))
                    results[m]["hits"] += hits

            report = f"--- BENCHMARK RESULTS (Queries: {num_queries}, K={k}) ---\n\n"
            report += f"Brute Force (Exact):\n  Avg Time: {bf_time/num_queries:.5f}s\n  Recall: 100%\n\n"
            
            for m in methods:
                avg_time = results[m]["time"] / num_queries
                recall = (results[m]["hits"] / (num_queries * k)) * 100
                speedup = (bf_time / results[m]["time"]) if results[m]["time"] > 0 else 0
                
                report += f"{m.upper()}:\n"
                report += f"  Avg Time: {avg_time:.5f}s (Speedup: {speedup:.1f}x)\n"
                report += f"  Recall:   {recall:.1f}%\n\n"
                
            return report