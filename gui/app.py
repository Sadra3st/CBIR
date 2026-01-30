import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import sys

# Ensure parent directory is in path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_db.crud import ImageRetriever

class CBIRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CBIR System - ResNet18 & LSH")
        self.root.geometry("1100x700")

        # Initialize Logic
        # Ensure 'data' directory exists
        if not os.path.exists("data"):
            os.makedirs("data")
        self.retriever = ImageRetriever(vector_db_path="data/vectors.npy", meta_db_path="data/metadata.json")
        
        self.query_image_path = None
        
        self._setup_ui()

    def _setup_ui(self):
        # --- Top Control Panel ---
        control_frame = tk.Frame(self.root, padx=10, pady=10, bg="#f0f0f0")
        control_frame.pack(fill=tk.X)

        tk.Label(control_frame, text="Search Method:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        
        self.method_var = tk.StringVar(value="brute_force")
        ttk.Combobox(control_frame, textvariable=self.method_var, 
                     values=["brute_force", "lsh"], state="readonly").pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="Load/Select Query Image", command=self.load_image).pack(side=tk.LEFT, padx=20)
        tk.Button(control_frame, text="Search", command=self.search, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)

        # --- Main Content Area ---
        content_frame = tk.Frame(self.root, padx=10, pady=10)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Query Image
        self.left_panel = tk.Frame(content_frame, width=300, borderwidth=2, relief="groove")
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        tk.Label(self.left_panel, text="Query Image", font=("Arial", 12, "bold")).pack(pady=5)
        self.query_label = tk.Label(self.left_panel, text="No image selected")
        self.query_label.pack(pady=20)

        # Right: Results
        self.right_panel = tk.Frame(content_frame)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(self.right_panel, text="Search Results", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Scrollable Frame for Results
        self.canvas = tk.Canvas(self.right_panel)
        self.scrollbar = ttk.Scrollbar(self.right_panel, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.query_image_path = file_path
            self.display_image(file_path, self.query_label, size=(250, 250))

    def display_image(self, path, label_widget, size=(150, 150)):
        try:
            img = Image.open(path)
            img.thumbnail(size)
            img_tk = ImageTk.PhotoImage(img)
            label_widget.config(image=img_tk, text="")
            label_widget.image = img_tk # Keep reference
        except Exception as e:
            label_widget.config(text="Error loading image", image="")

    def search(self):
        if not self.query_image_path:
            messagebox.showwarning("Warning", "Please select a query image first.")
            return

        method = self.method_var.get()
        results = self.retriever.search(self.query_image_path, k=10, method=method)

        # Clear previous results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        if not results:
            tk.Label(self.scrollable_frame, text="No results found.").pack(pady=20)
            return

        # Grid Layout for results
        row = 0
        col = 0
        max_cols = 3

        for res in results:
            frame = tk.Frame(self.scrollable_frame, borderwidth=1, relief="solid", padx=5, pady=5)
            frame.grid(row=row, column=col, padx=10, pady=10)

            # Image
            lbl_img = tk.Label(frame)
            lbl_img.pack()
            if os.path.exists(res['path']):
                self.display_image(res['path'], lbl_img)
            else:
                lbl_img.config(text="Img not found")

            # Info
            score_text = f"Dist: {res['score']:.4f}"
            tk.Label(frame, text=score_text).pack()
            tk.Label(frame, text=f"Cat: {res['category']}", font=("Arial", 8)).pack()

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

if __name__ == "__main__":
    root = tk.Tk()
    app = CBIRApp(root)
    root.mainloop()