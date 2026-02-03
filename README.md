## 🏞️ CBIR – Content-Based Image Retrieval System
A full-featured Content-Based Image Retrieval (CBIR) system implemented in Python, combining deep learning–based image embeddings with multiple exact and approximate nearest-neighbor search algorithms.
The project includes a custom vector database, concurrent indexing, benchmarking tools, and an interactive graphical user interface.

## 📌 Overview
This project enables image similarity search by converting images into high-dimensional feature vectors using a deep neural network and retrieving visually similar images based on distance metrics.
It is designed to be modular, extensible, and research-friendly, making it suitable for experimentation with approximate nearest neighbor (ANN) algorithms.

## 📝 How It Works

1. **Feature Extraction**

    * Images are processed with a pre-trained **ResNet18** model.
    * Each image is converted into a **512-dimensional** vector, capturing its visual features.

2. **Vector Database**

    * Vectors are stored in a **NumPy (.npy) file**, and metadata (categories, IDs, etc.) is stored in JSON.
    * The database is **thread-safe** using **Reentrant Locks (RLock)**, allowing safe concurrent read/write operations.

3. **Indexing**

    * Heavy indexing operations (graph/tree construction) are executed asynchronously in background threads to avoid freezing the GUI.
    * Supports multiple approximate nearest-neighbor search structures for fast retrieval.

4. **Searching**

    * Exact Search: **Brute-force k-NN** using Euclidean distance.
    * Approximate Search:
        * **Locality Sensitive Hashing (LSH)** – hash-based similarity.
        * **Navigable Small World (NSW)** – graph-based greedy search.
        * **Annoy-style Trees** – random projection forest.

5. **Analytics & Visualization**

    * **t-SNE** reduces the 512-dimensional space to 2D for visualization.
    * Optimized for large datasets (9,000+ images) with PCA pre-reduction and random sampling.
    * Benchmarking tools compare recall (accuracy) and query time of all algorithms.

6. **Graphical User Interface (GUI)**

* Drag-and-drop image querying.
* Dropdown to select search algorithms on the fly.
* Category explorer for filtering and sampling images.
* Displays search results with similarity scores, categories, and previews (using thumbnails if originals are missing).



## ✨ Key Features
### 1. Core Architecture

* **Vector Database**
    * Custom persistence layer
    * Image feature vectors stored as NumPy (`.npy`) files
    * Metadata stored in structured JSON format

* **Deep Learning Embeddings**
    * Uses a pre-trained **ResNet18** model
    * Extracts **512-dimensional feature vectors** from images

* **Concurrency & Atomicity**
    * Thread-safe implementation using **Reentrant Locks (RLock)**
    * Ensures consistency during simultaneous read/write operations
    
* **Asynchronous Indexing**
    * Heavy indexing tasks (graph and tree construction) run in **background threads**
    * Prevents GUI freezing during large database updates
---
### 2. Data Management (CRUD)

* **Create**
    * Add individual images to the database
    * Support for custom user-defined categories

* **Read**
    * Retrieve image vectors and metadata using unique IDs

* **Update**
    * Remove images and their associated index entries

* **Reset**
    * One-click option to securely wipe the entire database and all indexes

* **Batch Import**
    * Specialized importer for the **Caltech-101 dataset**
    * Uses pre-computed embeddings
    * Automatically generates thumbnails for GUI visualization

## 3. Search Algorithms 

The system supports both **exact** and **approximate** similarity search methods:

* **Exact Search**
    * Brute-force **k-Nearest Neighbors (k-NN)**
    * Euclidean distance metric

* **Approximate Search**
    * **Locality Sensitive Hashing (LSH)**
        * Random Hyperplane projection
    * **Navigable Small World (NSW) Graph**
        * Greedy graph traversal
    * **Annoy-style Trees**
        * Random Projection Forests for tree-based ANN search

Each algorithm can be selected dynamically from the GUI.

## 4. Analytics & Visualization

* **Dimensionality Reduction**
    * Visualizes the 512-D embedding space in **2D**
    * Uses **t-SNE** with **Cosine distance**

* **Performance Optimization**
    * PCA pre-reduction
    * Random sampling for large datasets (9,000+ images)

* **Benchmarking Suite**
    * Automated evaluation tool
    * Compares:
        * **Recall (accuracy)**
        * **Query time (speed)**
    * Benchmarks all ANN algorithms against the brute-force baseline

## 5. Graphical User Interface (GUI)

* **Image Query**
    * Load query images via file selection (drag-and-drop style workflow)

* **Algorithm Selection**
    * Dropdown menu to switch between:
        * Brute Force
        * LSH
        * NSW
        * Annoy-style Trees

* **Category Explorer**
    * Filter images by category
    * Random sampling from selected classes

* **Visual Feedback**
    * Displays:
        * Retrieved images
        * Similarity scores
        * Category labels
    * Uses stored thumbnails if original image files are unavailable



## 🏗️ Project Structure
```
CBIR/
├── database/          # Vector storage & metadata
├── embedding/         # ResNet18 feature extraction
├── gui/               # Graphical user interface
├── knn/               # Exact k-NN search
├── lsh/               # LSH approximate search
├── nsw/               # Graph-based NSW search
├── benchmarks/        # Evaluation & benchmarking tools
├── vector_db/         # Vector database implementation
├── import.py          # Dataset import utilities
└── main.py            # Application entry point


```

## 📦 Requirements & Setup
* Python 3.7+

1. **Install libraries**:
```
pip install torch torchvision numpy pillow matplotlib seaborn scikit-learn tqdm

```
2. **Clone the Project**:
```
git clone git@github.com:MohammadDaeizadeh/CBIR.git
```
3. **Add your images in `caltech101` directory**

4. **Run import.py**:
```
python import.py
```
5. **Run app.py**:
```
python app.py
```


## 👨‍💻 Contributors
* **Sadra Seyedtabaei** - GUI & Additional Features
* **Mohammad Daeizadeh** – Main Features & Documentes

## ✅ Acknowledgments
* **Dr. Ali Katanforoosh** – Instructor, SBU