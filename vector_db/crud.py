import os
import uuid
from vector_db.storage import VectorDB
from embedding.resnet import load_resnet18, image_to_embedding
from lsh.lsh import LSH
from knn.brute_force import BruteForceSearch

class ImageRetriever:
    def __init__(self, vector_db_path="data/vectors.npy", meta_db_path="data/metadata.json"):
        self.db = VectorDB(vector_db_path, meta_db_path)
        self.model = load_resnet18()
        self.bf_search = BruteForceSearch(metric="euclidean")
        self.lsh = LSH(dim=512, num_bits=10, num_tables=2)
        
        # Rebuild LSH index from existing DB data
        self.refresh_lsh_index()

    def refresh_lsh_index(self):
        print("Building LSH Index...")
        all_vectors = self.db.get_all_vectors()
        if all_vectors:
            self.lsh.index(all_vectors)
        print(f"LSH Index built with {len(all_vectors)} vectors.")

    def add_image(self, image_path, category="unknown"):
        """Process an image, generate embedding, and save to DB."""
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return False

        try:
            # Generate Embedding
            embedding = image_to_embedding(self.model, image_path)
            
            # Generate ID
            img_id = str(uuid.uuid4())
            
            # Create Metadata
            meta = {
                "path": image_path,
                "category": category,
                "filename": os.path.basename(image_path)
            }

            # Insert into DB
            self.db.insert(img_id, embedding, meta)
            
            # Save to disk
            self.db.save()
        
            return img_id
        except Exception as e:
            print(f"Failed to add image: {e}")
            return None

    def search(self, query_image_path, k=5, method="brute_force"):
        """
        Search for similar images.
        method: 'brute_force' or 'lsh'
        """
        # 1 Convert Query Image to Vector
        query_vec = image_to_embedding(self.model, query_image_path)

        # 2 Search
        results = []
        vectors = self.db.get_all_vectors()

        if method == "lsh":
            self.lsh.index(vectors) 
            results = self.lsh.query(query_vec, vectors, k=k)
        else:
            results = self.bf_search.search(query_vec, vectors, k=k)

        # 3 Enrich results
        enriched_results = []
        for img_id, score in results:
            meta = self.db.get_metadata(img_id)
            if meta:
                enriched_results.append({
                    "id": img_id,
                    "score": score,
                    "path": meta.get("path"),
                    "category": meta.get("category")
                })
        
        return enriched_results