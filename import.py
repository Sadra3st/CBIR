import numpy as np
import os
import sys
import base64
import io
from PIL import Image

# ensure we can import from vector_db
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vector_db.crud import ImageRetriever
except ImportError:
    print("Error: Could not import ImageRetriever.")
    sys.exit(1)

def generate_thumbnail(path):
    """Try to load image from path and create base64 thumbnail"""
    # fix paths for current OS
    path = path.replace('\\', '/')
    
    # check if file exists relative to current script
    if not os.path.exists(path):
        # try stripping leading slash or folder if needed
        # assuming folder structure: current_dir/caltech101/...
        if os.path.exists(os.path.join(".", path)):
            path = os.path.join(".", path)
        else:
            return None, path

    try:
        with Image.open(path) as img:
            img.thumbnail((200, 200))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=70)
            return base64.b64encode(buffered.getvalue()).decode('utf-8'), path
    except:
        return None, path

def load_and_import():
    emb_path = "caltech101_embeddings.npy"
    ids_path = "caltech101_image_ids.npy"
    
    if not os.path.exists(emb_path) or not os.path.exists(ids_path):
        print(f"Error: Could not find {emb_path} or {ids_path}.")
        return

    print("Loading NumPy arrays...")
    try:
        embeddings = np.load(emb_path)
        image_paths = np.load(ids_path)
    except Exception as e:
        print(f"Error loading arrays: {e}")
        return

    print(f"Loaded {len(embeddings)} vectors. Processing images & thumbnails...")
    print("Note: This might take a moment to generate thumbnails for the UI.")

    cleaned_paths = []
    categories = []
    thumbnails = []

    for i, raw_path in enumerate(image_paths):
        path_str = str(raw_path)
        
        # generate thumbnail
        thumb, valid_path = generate_thumbnail(path_str)
        thumbnails.append(thumb)
        cleaned_paths.append(valid_path)
        
        # extract category
        # expected: caltech101/category/image.jpg
        parts = valid_path.replace('\\', '/').split('/')
        if len(parts) >= 2:
            categories.append(parts[-2])
        else:
            categories.append("unknown")
            
        if (i + 1) % 500 == 0:
            print(f"Prepared {i + 1} images...")

    print("Initializing Database...")
    retriever = ImageRetriever()

    # Reset DB first to avoid duplicates or conflicts
    retriever.reset_database()

    print("Importing to DB...")
    retriever.import_batch(embeddings, cleaned_paths, categories, thumbnails)

if __name__ == "__main__":
    load_and_import()