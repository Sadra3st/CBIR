import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk, ImageDraw
import os
import sys
import numpy as np
import base64
import io
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from vector_db.crud import ImageRetriever
except ImportError:
    ImageRetriever = None

THEME = {
    "bg_main":   "#18181b",  # Background
    "bg_sec":    "#27272a",  # Sidebar/Header
    "bg_thrd":   "#3f3f46",  # Borders/Inputs
    "bg_hover":  "#52525b",  # Hover state
    "accent":    "#e11d48",  
    "accent_hov": "#fb7185", # Hover
    "text_main": "#f4f4f5",  
    "text_dim":  "#a1a1aa",  
    "success":   "#10b981",  
    "font_ui":   ("Segoe UI", 10),
    "font_head": ("Segoe UI", 14, "bold"),
}

class ModernButton(tk.Button):
    """Custom Flat Button"""
    def __init__(self, master, **kwargs):
        self.default_bg = kwargs.get('bg', THEME['bg_sec'])
        self.hover_bg = kwargs.pop('hover_bg', THEME['bg_hover'])
        self.default_fg = kwargs.get('fg', THEME['text_main'])
        
        kwargs['bg'] = self.default_bg
        kwargs['fg'] = self.default_fg
        kwargs['relief'] = tk.FLAT
        kwargs['bd'] = 0
        kwargs['highlightthickness'] = 0
        kwargs['font'] = kwargs.get('font', ("Segoe UI", 10))
        kwargs['cursor'] = "hand2"
        kwargs['padx'] = 20
        kwargs['pady'] = 8
        kwargs['activebackground'] = self.hover_bg
        kwargs['activeforeground'] = self.default_fg
        
        super().__init__(master, **kwargs)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        if self['state'] != 'disabled':
            self['bg'] = self.hover_bg

    def on_leave(self, e):
        if self['state'] != 'disabled':
            self['bg'] = self.default_bg

class ActionButton(ModernButton):
    """Highlighted Action Button"""
    def __init__(self, master, **kwargs):
        kwargs['bg'] = THEME['accent']
        kwargs['hover_bg'] = THEME['accent_hov']
        kwargs['fg'] = "white"
        kwargs['font'] = ("Segoe UI", 10, "bold")
        super().__init__(master, **kwargs)

class CBIRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CBIR System")
        self.root.geometry("1300x900")
        self.root.configure(bg=THEME['bg_main'])
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if ImageRetriever:
            self.retriever = ImageRetriever(vector_db_path="data/vectors.npy", meta_db_path="data/metadata.json")
        else:
            self.retriever = None
        
        self.query_image_path = None
        self._setup_matplotlib_style()
        self._setup_styles()
        self._setup_ui()
        self._check_status()

    def _check_status(self):
        if self.retriever:
            status = self.retriever.indexing_status
            if status != "Ready":
                self.status_label.config(text=f"System: {status}", fg=THEME['accent'])
                # Check again in 1 second
                self.root.after(1000, self._check_status)
            else:
                self.status_label.config(text="Ready", fg=THEME['success'])

    def on_close(self):
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    def _setup_matplotlib_style(self):
        plt.style.use('dark_background')
        plt.rcParams.update({
            "axes.facecolor": THEME['bg_main'],
            "figure.facecolor": THEME['bg_main'],
            "savefig.facecolor": THEME['bg_main'],
            "text.color": THEME['text_dim'],
            "axes.labelcolor": THEME['text_dim'],
            "xtick.color": THEME['text_dim'],
            "ytick.color": THEME['text_dim'],
            "grid.color": THEME['bg_thrd'],
            "axes.edgecolor": THEME['bg_thrd']
        })

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.map('TCombobox', fieldbackground=[('readonly', THEME['bg_thrd'])],
                                selectbackground=[('readonly', THEME['bg_hover'])],
                                selectforeground=[('readonly', "white")])
        style.configure('TCombobox', background=THEME['bg_sec'], foreground=THEME['text_main'],
                        arrowcolor=THEME['text_main'], fieldbackground=THEME['bg_thrd'], borderwidth=0)
        style.configure("Vertical.TScrollbar", gripcount=0,
                        background=THEME['bg_thrd'], darkcolor=THEME['bg_main'], lightcolor=THEME['bg_main'],
                        troughcolor=THEME['bg_main'], bordercolor=THEME['bg_main'], arrowcolor=THEME['text_dim'])
        style.configure("TScale", background=THEME['bg_sec'], troughcolor=THEME['bg_thrd'], sliderthickness=15)

    def _setup_ui(self):
        header_container = tk.Frame(self.root, bg=THEME['bg_sec'])
        header_container.pack(fill=tk.X)
        header = tk.Frame(header_container, bg=THEME['bg_sec'], height=70, padx=30)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title_frame = tk.Frame(header, bg=THEME['bg_sec'])
        title_frame.pack(side=tk.LEFT, pady=10)
        tk.Label(title_frame, text="🖼️", font=("Segoe UI Emoji", 24), bg=THEME['bg_sec'], fg=THEME['accent']).pack(side=tk.LEFT)
        tk.Label(title_frame, text="CBIR System", font=("Segoe UI", 18, "bold"), bg=THEME['bg_sec'], fg="white").pack(side=tk.LEFT, padx=10)
        tk.Label(title_frame, text="|  Vector Search Engine", font=("Segoe UI", 11), bg=THEME['bg_sec'], fg=THEME['text_dim']).pack(side=tk.LEFT, padx=(5,0), pady=(4,0))
        tk.Label(title_frame, text="Made by Sadra Seyedtabaei & Mohammad Daeizadeh", font=("Segoe UI", 10, "italic"), bg=THEME['bg_sec'], fg=THEME['text_dim']).pack(side=tk.LEFT, padx=20, pady=(5,0))

        nav_frame = tk.Frame(header, bg=THEME['bg_sec'])
        nav_frame.pack(side=tk.RIGHT)
        self.btn_tab_search = tk.Button(nav_frame, text="🔍 Search", command=lambda: self.switch_view("search"), font=("Segoe UI", 10, "bold"), bd=0, relief=tk.FLAT, padx=20, pady=8, cursor="hand2")
        self.btn_tab_search.pack(side=tk.LEFT, padx=5)
        self.btn_tab_viz = tk.Button(nav_frame, text="📊 Analytics", command=lambda: self.switch_view("viz"), font=("Segoe UI", 10, "bold"), bd=0, relief=tk.FLAT, padx=20, pady=8, cursor="hand2")
        self.btn_tab_viz.pack(side=tk.LEFT, padx=5)
        tk.Frame(header_container, bg=THEME['bg_thrd'], height=1).pack(fill=tk.X)

        self.content_area = tk.Frame(self.root, bg=THEME['bg_main'])
        self.content_area.pack(fill=tk.BOTH, expand=True)
        self.view_search = tk.Frame(self.content_area, bg=THEME['bg_main'])
        self.view_viz = tk.Frame(self.content_area, bg=THEME['bg_main'])
        self._build_search_view()
        self._build_viz_view()
        self.switch_view("search")

    def switch_view(self, view_name):
        default_style = {"bg": THEME['bg_sec'], "fg": THEME['text_dim']}
        active_style = {"bg": THEME['bg_thrd'], "fg": "white"}
        self.btn_tab_search.config(**default_style)
        self.btn_tab_viz.config(**default_style)
        self.view_search.pack_forget()
        self.view_viz.pack_forget()
        if view_name == "search":
            self.view_search.pack(fill=tk.BOTH, expand=True)
            self.btn_tab_search.config(**active_style)
        else:
            self.view_viz.pack(fill=tk.BOTH, expand=True)
            self.btn_tab_viz.config(**active_style)

    def _add_separator(self, parent):
        tk.Frame(parent, bg=THEME['bg_thrd'], height=1).pack(fill=tk.X, pady=10)

    def _build_search_view(self):
        container = tk.Frame(self.view_search, bg=THEME['bg_main'])
        container.pack(fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(container, bg=THEME['bg_sec'], width=300, padx=20, pady=20)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # SETTINGS
        tk.Label(sidebar, text="SETTINGS", font=("Segoe UI", 9, "bold"), bg=THEME['bg_sec'], fg=THEME['text_dim']).pack(anchor="w", pady=(0,10))
        tk.Label(sidebar, text="Algorithm Strategy", font=("Segoe UI", 9), bg=THEME['bg_sec'], fg="white").pack(anchor="w")
        
        self.method_var = tk.StringVar(value="Brute Force")
        ttk.Combobox(sidebar, textvariable=self.method_var, 
                     values=["Brute Force", "LSH (Approximate)", "NSW (Graph-Based)", "Annoy (Tree-Based)"], 
                     state="readonly", font=("Segoe UI", 9)).pack(fill=tk.X, pady=(5, 15))

        tk.Label(sidebar, text=f"Top K Results", font=("Segoe UI", 9), bg=THEME['bg_sec'], fg="white").pack(anchor="w")
        scale_frame = tk.Frame(sidebar, bg=THEME['bg_sec'])
        scale_frame.pack(fill=tk.X, pady=(5, 0))
        self.k_var = tk.IntVar(value=5)
        tk.Scale(scale_frame, from_=1, to=20, orient=tk.HORIZONTAL, variable=self.k_var, bg=THEME['bg_sec'], fg=THEME['text_main'], showvalue=1, troughcolor=THEME['bg_thrd'], activebackground=THEME['accent'], highlightthickness=0, bd=0, length=200).pack(fill=tk.X)
        self._add_separator(sidebar)

        # EXPLORE
        tk.Label(sidebar, text="EXPLORE", font=("Segoe UI", 9, "bold"), bg=THEME['bg_sec'], fg=THEME['text_dim']).pack(anchor="w", pady=(10,10))
        self.category_var = tk.StringVar()
        self.cat_combo = ttk.Combobox(sidebar, textvariable=self.category_var, state="readonly", font=("Segoe UI", 9))
        self.cat_combo.pack(fill=tk.X, pady=(0, 10))
        self._refresh_category_list()
        ModernButton(sidebar, text="🎲  Sample Random", command=self.search_by_category, bg=THEME['bg_thrd']).pack(fill=tk.X)
        self._add_separator(sidebar)

        # ACTIONS
        tk.Label(sidebar, text="ACTIONS", font=("Segoe UI", 9, "bold"), bg=THEME['bg_sec'], fg=THEME['text_dim']).pack(anchor="w", pady=(10,10))
        btn_grid = tk.Frame(sidebar, bg=THEME['bg_sec'])
        btn_grid.pack(fill=tk.X)
        ModernButton(btn_grid, text="📂 Load", command=self.load_image, bg=THEME['bg_thrd']).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ModernButton(btn_grid, text="➕ Add DB", command=self.add_to_database, bg=THEME['bg_thrd']).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Reset
        btn_reset_frame = tk.Frame(sidebar, bg=THEME['bg_sec'])
        btn_reset_frame.pack(fill=tk.X, pady=(10,0))
        ModernButton(btn_reset_frame, text="🗑️ Reset DB", command=self.reset_db, bg=THEME['bg_thrd'], fg="#ef4444").pack(fill=tk.X)

        ActionButton(sidebar, text="🚀  Find Similar Images", command=self.search).pack(fill=tk.X, pady=(15, 0))

        # PREVIEW
        tk.Frame(sidebar, height=20, bg=THEME['bg_sec']).pack() 
        tk.Label(sidebar, text="QUERY PREVIEW", font=("Segoe UI", 8, "bold"), bg=THEME['bg_sec'], fg=THEME['text_dim']).pack(anchor="w", pady=(0,5))
        self.query_frame = tk.Frame(sidebar, bg=THEME['bg_thrd'], height=200, highlightthickness=1, highlightbackground=THEME['bg_hover'])
        self.query_frame.pack(fill=tk.X)
        self.query_frame.pack_propagate(False)
        self.query_label = tk.Label(self.query_frame, text="No Image Selected", bg=THEME['bg_thrd'], fg=THEME['text_dim'])
        self.query_label.pack(expand=True)

        self.status_label = tk.Label(sidebar, text="Ready", font=("Segoe UI", 9), bg=THEME['bg_sec'], fg=THEME['text_dim'])
        self.status_label.pack(side=tk.BOTTOM, pady=20)

        # RESULTS 
        res_area = tk.Frame(container, bg=THEME['bg_main'], padx=30, pady=20)
        res_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        tk.Label(res_area, text="Search Results", font=THEME['font_head'], bg=THEME['bg_main'], fg=THEME['text_main']).pack(anchor="w", pady=(0,15))
        cv_cont = tk.Frame(res_area, bg=THEME['bg_main'])
        cv_cont.pack(fill=tk.BOTH, expand=True)
        self.results_canvas = tk.Canvas(cv_cont, bg=THEME['bg_main'], highlightthickness=0)
        sb = ttk.Scrollbar(cv_cont, orient="vertical", command=self.results_canvas.yview)
        self.results_frame = tk.Frame(self.results_canvas, bg=THEME['bg_main'])
        self.results_frame.bind("<Configure>", lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all")))
        self.results_canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.results_canvas.bind_all("<MouseWheel>", lambda e: self.results_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

    def _build_viz_view(self):
        top = tk.Frame(self.view_viz, bg=THEME['bg_main'], height=80, padx=30, pady=30)
        top.pack(fill=tk.X)
        tk.Label(top, text="Analytics Dashboard", font=THEME['font_head'], bg=THEME['bg_main'], fg=THEME['text_main']).pack(side=tk.LEFT)
        
        # Benchmark
        ModernButton(top, text="⏱️ Run Benchmark", command=self.run_benchmark, bg=THEME['accent'], hover_bg=THEME['accent_hov'], padx=15, pady=5).pack(side=tk.RIGHT, padx=10)
        
        ModernButton(top, text="🔄 Refresh Chart", command=self.visualize_db, bg=THEME['bg_thrd'], padx=15, pady=5).pack(side=tk.RIGHT)
        
        self.plot_container = tk.Frame(self.view_viz, bg=THEME['bg_main'])
        self.plot_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        tk.Label(self.plot_container, text="Use controls above to view data", bg=THEME['bg_main'], fg=THEME['text_dim'], font=("Segoe UI", 12)).pack(expand=True)

    def run_benchmark(self):
        if not self.retriever: return
        self.status_label.config(text="Running...", fg=THEME['accent'])
        self.root.update()
        
        try:
            report = self.retriever.benchmark_algorithms(num_queries=50, k=10)
            
            # Show results 
            top = tk.Toplevel(self.root)
            top.title("Benchmark Results")
            top.geometry("600x500")
            top.configure(bg=THEME['bg_main'])
            
            text = tk.Text(top, bg=THEME['bg_sec'], fg=THEME['text_main'], font=("Consolas", 10), padx=20, pady=20, bd=0)
            text.pack(fill=tk.BOTH, expand=True)
            text.insert(tk.END, report)
            text.config(state=tk.DISABLED)
            
            self.status_label.config(text="Benchmark Complete", fg=THEME['success'])
        except Exception as e:
            messagebox.showerror("Error", f"failed: {e}")
            self.status_label.config(text="Error", fg="red")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.query_image_path = file_path
            self.display_image(file_path, self.query_label, size=(280, 190), bg_color=THEME['bg_thrd'])
            self.status_label.config(text="Image Loaded", fg=THEME['success'])

    def add_to_database(self):
        if not self.retriever or not self.query_image_path:
            messagebox.showwarning("CBIR", "Load an image first.")
            return
        category = simpledialog.askstring("Add Image", "Enter category/label for this image:", parent=self.root)
        if category is None: return 
        if not category.strip(): category = "user_added"
        try:
            self.status_label.config(text="Adding...", fg=THEME['accent'])
            self.root.update()
            img_id = self.retriever.add_image(self.query_image_path, category=category)
            if img_id:
                messagebox.showinfo("Success", f"Image added successfully!\nCategory: {category}")
                self.status_label.config(text="Image Added to DataBase", fg=THEME['success'])
                self._refresh_category_list()
            else:
                messagebox.showerror("Error", "Failed to add image!")
                self.status_label.config(text="Add Failed", fg="red")
        except Exception as e:
            messagebox.showerror("Error", f"{str(e)}")
            self.status_label.config(text="Error", fg="red")

    def reset_db(self):
        if not self.retriever: return
        if messagebox.askyesno("Reset Database", "Are you sure? This cannot be undone."):
            self.retriever.reset_database()
            self.status_label.config(text="Database Cleared", fg=THEME['accent'])
            self._refresh_category_list()
            for w in self.results_frame.winfo_children(): w.destroy()
            self.results_canvas.yview_moveto(0)

    def _refresh_category_list(self):
        if not self.retriever: return
        try:
            meta = self.retriever.db.metadata
            cats = sorted(list(set(m.get('category', 'Unknown') for m in meta.values())))
            self.cat_combo['values'] = cats
            if cats:
                self.cat_combo.current(0)
            else:
                self.cat_combo.set('')
        except Exception as e:
            print(f"Error refreshing categories: {e}")

    def search_by_category(self):
        if not self.retriever: return
        cat = self.category_var.get()
        if not cat: 
            messagebox.showwarning("CBIR", "Please select a category first.")
            return
        try:
            meta = self.retriever.db.metadata
            candidates = [k for k, v in meta.items() if v.get('category') == cat]
            if not candidates:
                messagebox.showinfo("Info", "No images found in this category.")
                return
            random.shuffle(candidates)
            found = False
            for img_id in candidates:
                path = meta[img_id].get('path')
                if path and os.path.exists(path):
                    self.query_image_path = path
                    self.display_image(path, self.query_label, size=(280, 190), bg_color=THEME['bg_thrd'])
                    self.status_label.config(text=f"Sample: {os.path.basename(path)}", fg=THEME['success'])
                    self.search()
                    found = True
                    break
            if not found:
                messagebox.showwarning("Error", "The original images for this category seem to be missing from the disk.")
        except Exception as e:
             messagebox.showerror("Error", f"Category search failed: {e}")

    def display_image(self, path, label_widget, size=(200, 200), bg_color=THEME['bg_sec']):
        try:
            img = Image.open(path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            label_widget.config(image=img_tk, text="", bg=bg_color)
            label_widget.image = img_tk
        except Exception:
            label_widget.config(text="Error", image="", bg=bg_color)

    def display_base64(self, b64_data, label_widget, size=(150, 150), bg_color=THEME['bg_thrd']):
        try:
            img_data = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_data))
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            label_widget.config(image=img_tk, text="", bg=bg_color)
            label_widget.image = img_tk
        except Exception:
            label_widget.config(text="Err", image="", bg=bg_color)

    def search(self):
        if not self.retriever or not self.query_image_path:
            messagebox.showwarning("CBIR", "Please load an image first.")
            return
        
        # select algo
        algo_name = self.method_var.get()
        method_map = {
            "Brute Force": "brute_force",
            "LSH (Approximate)": "lsh",
            "NSW (Graph-Based)": "nsw",
            "Annoy (Tree-Based)": "annoy"
        }
        method = method_map.get(algo_name, "brute_force")
        
        k_count = self.k_var.get()
        self.status_label.config(text=f"Searching ({k_count})...", fg=THEME['accent'])
        self.root.update()
        try:
            results = self.retriever.search(self.query_image_path, k=k_count, method=method)
            for w in self.results_frame.winfo_children(): w.destroy()
            row, col = 0, 0
            width = self.results_canvas.winfo_width()
            max_cols = max(3, width // 230)
            TILE_DIM = 180
            for idx, res in enumerate(results):
                card = tk.Frame(self.results_frame, bg=THEME['bg_sec'])
                card.grid(row=row, column=col, padx=10, pady=10)
                img_f = tk.Frame(card, bg=THEME['bg_thrd'], height=TILE_DIM, width=TILE_DIM)
                img_f.pack()
                img_f.pack_propagate(False)
                lbl = tk.Label(img_f, bg=THEME['bg_thrd'])
                lbl.pack(expand=True) 
                if os.path.exists(res['path']):
                    self.display_image(res['path'], lbl, (TILE_DIM, TILE_DIM), THEME['bg_thrd'])
                elif res.get('thumbnail'):
                    self.display_base64(res['thumbnail'], lbl, (TILE_DIM, TILE_DIM), THEME['bg_thrd'])
                info = tk.Frame(card, bg=THEME['bg_sec'], padx=10, pady=10)
                info.pack(fill=tk.X)
                meta_frame = tk.Frame(info, bg=THEME['bg_sec'])
                meta_frame.pack(fill=tk.X)
                tk.Label(meta_frame, text=f"Match #{idx+1}", font=("Segoe UI", 9, "bold"), fg=THEME['accent'], bg=THEME['bg_sec']).pack(side=tk.LEFT)
                tk.Label(meta_frame, text=f"{res['score']:.3f}", font=("Segoe UI", 9), fg=THEME['text_dim'], bg=THEME['bg_sec']).pack(side=tk.RIGHT)
                tk.Label(info, text=res.get('category', 'Unknown'), font=("Segoe UI", 8), fg="white", bg=THEME['bg_sec']).pack(anchor="w", pady=(2,0))
                col += 1
                if col >= max_cols: col=0; row+=1
            self.status_label.config(text=f"Found {len(results)} matches", fg=THEME['success'])
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def visualize_db(self):
        if not self.retriever: return
        for widget in self.plot_container.winfo_children(): widget.destroy()

        try:
            ids, vectors, labels = self.retriever.get_all_embeddings_for_viz()
            n_samples = len(vectors)
            
            if n_samples < 3:
                tk.Label(self.plot_container, text="Insufficient data (need 3+ images)", bg=THEME['bg_main'], fg="red").pack()
                return

            MAX_SAMPLES = 800  
            
            if n_samples > MAX_SAMPLES:
                indices = np.random.choice(n_samples, MAX_SAMPLES, replace=False)
                vectors = vectors[indices]
                labels = [labels[i] for i in indices]
                n_samples = MAX_SAMPLES
                title_suffix = f" (Sample of {MAX_SAMPLES})"
            else:
                title_suffix = ""

            reduced_vecs = None
            title_text = "Dataset Projection"
            
            if n_samples > 5:
                if vectors.shape[1] > 50:
                    pca_prep = PCA(n_components=50)
                    vectors_pca = pca_prep.fit_transform(vectors)
                else:
                    vectors_pca = vectors

                perp = min(30, n_samples - 1)
                
                tsne = TSNE(
                    n_components=2, 
                    perplexity=perp, 
                    metric='cosine', 
                    init='pca', 
                    random_state=42
                )
                reduced_vecs = tsne.fit_transform(vectors_pca)
                title_text = f"Dataset Projection (t-SNE){title_suffix}"
            else:
                pca = PCA(n_components=2)
                reduced_vecs = pca.fit_transform(vectors)
                title_text = "Dataset Projection (PCA)"

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(title_text, color=THEME['text_main'], pad=20)
            unique_labels = sorted(list(set(labels)))
            cmap = plt.get_cmap("tab10") if len(unique_labels) <= 10 else plt.get_cmap("tab20")
            for i, label in enumerate(unique_labels):
                idxs = [j for j, l in enumerate(labels) if l == label]
                points = reduced_vecs[idxs]
                color = cmap(i % 20)
                ax.scatter(points[:, 0], points[:, 1], label=label, color=color, alpha=0.9, s=80, edgecolors='none')
            for spine in ax.spines.values(): spine.set_visible(False)
            legend = ax.legend(title="Categories", frameon=False, loc="best")
            plt.setp(legend.get_texts(), color=THEME['text_dim'])
            plt.setp(legend.get_title(), color=THEME['text_main'])
            ax.grid(True, linestyle=':', alpha=0.15, color="white")
            canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            tk.Label(self.plot_container, text=f"Error: {e}", bg=THEME['bg_main'], fg="red").pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = CBIRApp(root)
    root.mainloop()