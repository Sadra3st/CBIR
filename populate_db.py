import os
from vector_db.crud import ImageRetriever

def populate(dataset_path):
    retriever = ImageRetriever(vector_db_path="data/vectors.npy", meta_db_path="data/metadata.json")
    
    valid_extensions = {".jpg", ".jpeg", ".png"}
    count = 0

    print(f"Scanning {dataset_path}...")
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                full_path = os.path.join(root, file)

                category = os.path.basename(root)
                
                print(f"Indexing: {file} ({category})")
                retriever.add_image(full_path, category=category)
                count += 1
                
                if count % 10 == 0:
                    print(f"Processed {count} images...")

    print(f"Done! Successfully indexed {count} images.")

if __name__ == "__main__":

    DATASET_FOLDER = "images_folder" 
    
    if not os.path.exists(DATASET_FOLDER):
        print(f"Folder '{DATASET_FOLDER}' does not exist. Please create it or change the path in the script.")
    else:
        populate(DATASET_FOLDER)