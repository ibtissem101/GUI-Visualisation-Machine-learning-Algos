import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
import numpy as np
import threading

class HierarchicalPage(ctk.CTkFrame):
    def __init__(self, master, app_instance):
        super().__init__(master, fg_color="#F5F5F5", corner_radius=0)
        self.app = app_instance
        self.figure = None
        self.canvas = None
        self.is_running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header = ctk.CTkLabel(
            self,
            text="Hierarchical Clustering (AGNES)",
            font=("Segoe UI", 26, "bold"),
            text_color="#0F172A",
            anchor="w"
        )
        header.pack(padx=30, pady=(30, 5), anchor="w")
        
        subtitle = ctk.CTkLabel(
            self,
            text="Builds a hierarchy of clusters using agglomerative (bottom-up) approach.",
            font=("Segoe UI", 14),
            text_color="#64748B",
            anchor="w"
        )
        subtitle.pack(padx=30, pady=(0, 20), anchor="w")

        # View Switcher
        self.view_var = ctk.StringVar(value="Parameters")
        self.view_switcher = ctk.CTkSegmentedButton(
            self, 
            values=["Parameters", "Visualization", "Insights"],
            variable=self.view_var,
            command=self.switch_view,
            font=("Segoe UI", 12, "bold"),
            height=32
        )
        self.view_switcher.pack(padx=30, pady=(0, 20), anchor="w")
        
        # Content Area
        self.content_area = ctk.CTkFrame(self, fg_color="transparent")
        self.content_area.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        
        # --- Parameters View ---
        self.params_frame = ctk.CTkFrame(
            self.content_area, 
            fg_color="white", 
            corner_radius=8, 
            border_width=1, 
            border_color="#E2E8F0"
        )
        
        # Controls container
        controls_inner = ctk.CTkFrame(self.params_frame, fg_color="transparent")
        controls_inner.pack(fill="both", expand=True, padx=40, pady=40)
        
        ctk.CTkLabel(
            controls_inner, 
            text="Configuration", 
            font=("Segoe UI", 18, "bold"), 
            text_color="#1E293B",
            anchor="w"
        ).pack(fill="x", pady=(0, 20))
        
        # Grid layout
        grid_frame = ctk.CTkFrame(controls_inner, fg_color="transparent")
        grid_frame.pack(fill="x", pady=(0, 20))
        grid_frame.grid_columnconfigure(0, weight=1)
        grid_frame.grid_columnconfigure(1, weight=1)
        
        # Number of Clusters
        ctk.CTkLabel(grid_frame, text="Number of Clusters (k)", text_color="#475569", font=("Segoe UI", 12), anchor="w").grid(row=0, column=0, padx=10, pady=(0, 5), sticky="w")
        self.k_entry = ctk.CTkEntry(grid_frame, placeholder_text="Enter k value", height=36, font=("Segoe UI", 12))
        self.k_entry.insert(0, "3")
        self.k_entry.grid(row=1, column=0, padx=10, pady=(0, 20), sticky="ew")
        
        # Linkage
        ctk.CTkLabel(grid_frame, text="Linkage Method", text_color="#475569", font=("Segoe UI", 12), anchor="w").grid(row=0, column=1, padx=10, pady=(0, 5), sticky="w")
        self.linkage_var = ctk.StringVar(value="ward")
        self.linkage_menu = ctk.CTkOptionMenu(
            grid_frame,
            values=["ward", "complete", "average", "single"],
            variable=self.linkage_var,
            fg_color="#F1F5F9",
            text_color="#1E293B",
            button_color="#CBD5E1",
            button_hover_color="#94A3B8",
            height=36,
            font=("Segoe UI", 12)
        )
        self.linkage_menu.grid(row=1, column=1, padx=10, pady=(0, 20), sticky="ew")

        # Metric
        ctk.CTkLabel(grid_frame, text="Metric", text_color="#475569", font=("Segoe UI", 12), anchor="w").grid(row=2, column=0, padx=10, pady=(0, 5), sticky="w")
        self.metric_var = ctk.StringVar(value="euclidean")
        self.metric_menu = ctk.CTkOptionMenu(
            grid_frame,
            values=["euclidean", "l1", "l2", "manhattan", "cosine"],
            variable=self.metric_var,
            fg_color="#F1F5F9",
            text_color="#1E293B",
            button_color="#CBD5E1",
            button_hover_color="#94A3B8",
            height=36,
            font=("Segoe UI", 12)
        )
        self.metric_menu.grid(row=3, column=0, padx=10, pady=(0, 20), sticky="ew")
        
        # Feature Selection
        ctk.CTkLabel(
            controls_inner, 
            text="Feature Selection", 
            font=("Segoe UI", 16, "bold"), 
            text_color="#1E293B",
            anchor="w"
        ).pack(fill="x", pady=(10, 10))
        
        fs_frame = ctk.CTkFrame(controls_inner, fg_color="transparent")
        fs_frame.pack(fill="x", pady=(0, 20))
        fs_frame.grid_columnconfigure(0, weight=1)
        fs_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(fs_frame, text="X Axis:", text_color="#475569", font=("Segoe UI", 12), anchor="w").grid(row=0, column=0, padx=10, pady=(0, 5), sticky="w")
        self.x_axis_var = ctk.StringVar(value="")
        self.x_axis_menu = ctk.CTkOptionMenu(fs_frame, variable=self.x_axis_var, values=["Load Data First"])
        self.x_axis_menu.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        
        ctk.CTkLabel(fs_frame, text="Y Axis:", text_color="#475569", font=("Segoe UI", 12), anchor="w").grid(row=0, column=1, padx=10, pady=(0, 5), sticky="w")
        self.y_axis_var = ctk.StringVar(value="")
        self.y_axis_menu = ctk.CTkOptionMenu(fs_frame, variable=self.y_axis_var, values=["Load Data First"])
        self.y_axis_menu.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="ew")
        
        # Update features button (hidden, called automatically when data loaded)
        self.bind("<Visibility>", self.update_feature_options)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(controls_inner, mode="indeterminate")
        
        # Buttons
        btn_frame = ctk.CTkFrame(controls_inner, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(20, 0))
        
        self.run_btn = ctk.CTkButton(
            btn_frame,
            text="Run Clustering",
            command=self.run_clustering,
            font=("Segoe UI", 13, "bold"),
            height=44,
            fg_color="#2D5BFF",
            hover_color="#1E40AF"
        )
        self.run_btn.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.dendro_btn = ctk.CTkButton(
            btn_frame,
            text="Show Dendrogram",
            command=self.show_dendrogram,
            font=("Segoe UI", 13, "bold"),
            fg_color="#64748B",
            hover_color="#475569",
            height=44
        )
        self.dendro_btn.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
        self.status_label = ctk.CTkLabel(controls_inner, text="", text_color="#64748B", font=("Segoe UI", 11))
        self.status_label.pack(fill="x", pady=(10, 0))

        # --- Visualization View ---
        self.viz_frame = ctk.CTkFrame(
            self.content_area, 
            fg_color="white", 
            corner_radius=8, 
            border_width=1, 
            border_color="#E2E8F0"
        )
        # Placeholder for viz
        self.plot_placeholder()
        
        # --- Insights View ---
        self.insights_frame = ctk.CTkFrame(
            self.content_area, 
            fg_color="white", 
            corner_radius=8, 
            border_width=1, 
            border_color="#E2E8F0"
        )
        self.insights_text = ctk.CTkTextbox(self.insights_frame, font=("Segoe UI", 14), wrap="word", padx=20, pady=20)
        self.insights_text.pack(fill="both", expand=True, padx=2, pady=2)
        self.insights_text.insert("0.0", "Run the algorithm to generate insights.")
        self.insights_text.configure(state="disabled")

        # Initialize view
        self.switch_view("Parameters")

    def switch_view(self, view_name):
        self.params_frame.pack_forget()
        self.viz_frame.pack_forget()
        self.insights_frame.pack_forget()
        
        if view_name == "Parameters":
            self.params_frame.pack(fill="both", expand=True)
        elif view_name == "Visualization":
            self.viz_frame.pack(fill="both", expand=True)
        elif view_name == "Insights":
            self.insights_frame.pack(fill="both", expand=True)

    def plot_placeholder(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        fig = Figure(figsize=(6, 4.5), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Run clustering to see results', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='#64748B', fontsize=14)
        ax.set_facecolor('white')
        ax.axis('off')
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def update_feature_options(self, event=None):
        df = self.app.get_dataframe()
        if df is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                self.x_axis_menu.configure(values=numeric_cols)
                self.y_axis_menu.configure(values=numeric_cols)
                if not self.x_axis_var.get() or self.x_axis_var.get() not in numeric_cols:
                    self.x_axis_var.set(numeric_cols[0])
                if len(numeric_cols) > 1:
                    if not self.y_axis_var.get() or self.y_axis_var.get() not in numeric_cols:
                        self.y_axis_var.set(numeric_cols[1])
                else:
                    self.y_axis_var.set(numeric_cols[0])

    def get_data(self):
        df = self.app.get_dataframe()
        if df is None:
            tk.messagebox.showwarning("No Data", "Please load a dataset first from the Data Loader page.")
            return None, None
            
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            tk.messagebox.showwarning("Insufficient Data", "Dataset needs at least 2 numeric columns.")
            return None, None
            
        x_col = self.x_axis_var.get()
        y_col = self.y_axis_var.get()
        
        if not x_col or not y_col:
            self.update_feature_options()
            x_col = self.x_axis_var.get()
            y_col = self.y_axis_var.get()
            
        return numeric_df[[x_col, y_col]].values, [x_col, y_col]

    def run_clustering(self):
        if self.is_running:
            return
            
        X, feature_names = self.get_data()
        if X is None: return
        
        try:
            k = int(self.k_entry.get())
            if k < 2:
                tk.messagebox.showerror("Invalid Input", "Number of clusters must be at least 2.")
                return
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Number of clusters must be an integer.")
            return
            
        linkage_method = self.linkage_var.get()
        metric = self.metric_var.get()
        
        if linkage_method == 'ward' and metric != 'euclidean':
             tk.messagebox.showwarning("Invalid Combination", "Ward linkage only supports Euclidean metric. Using Euclidean.")
             metric = 'euclidean'
        
        # Show progress
        self.is_running = True
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.start()
        self.status_label.configure(text="Running Hierarchical clustering...")
        self.status_label.pack(fill="x", pady=(0, 10))
        self.run_btn.configure(state="disabled", text="Running...")
        self.dendro_btn.configure(state="disabled")
        
        # Run in thread
        thread = threading.Thread(target=self._run_clustering_thread, args=(X, k, linkage_method, metric, feature_names))
        thread.daemon = True
        thread.start()

    def _run_clustering_thread(self, X, k, linkage_method, metric, feature_names):
        try:
            # Sample if large
            if len(X) > 3000:  # Lower threshold for hierarchical
                sample_idx = np.random.choice(len(X), 3000, replace=False)
                X_sample = X[sample_idx]
            else:
                X_sample = X
            
            # Use metric parameter (compatible with newer sklearn)
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method, metric=metric)
            labels = model.fit_predict(X_sample)
            
            # Calculate Silhouette Score
            if len(set(labels)) > 1:
                sil_score = silhouette_score(X_sample, labels)
            else:
                sil_score = -1.0
            
            self.after(0, lambda: self._finish_clustering(
                X_sample.copy(), 
                labels.copy(), 
                list(feature_names), 
                linkage_method,
                sil_score
            ))
        except Exception as e:
            # Fallback for older sklearn versions if 'metric' fails (try 'affinity')
            if "unexpected keyword argument 'metric'" in str(e):
                try:
                    model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method, affinity=metric)
                    labels = model.fit_predict(X_sample)
                    
                    if len(set(labels)) > 1:
                        sil_score = silhouette_score(X_sample, labels)
                    else:
                        sil_score = -1.0
                        
                    self.after(0, lambda: self._finish_clustering(
                        X_sample.copy(), 
                        labels.copy(), 
                        list(feature_names), 
                        linkage_method,
                        sil_score
                    ))
                    return
                except Exception as e2:
                    self.after(0, lambda err=str(e2): self._handle_error(err))
            else:
                self.after(0, lambda err=str(e): self._handle_error(err))

    def _finish_clustering(self, X, labels, feature_names, linkage_method, sil_score):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.configure(text="✓ Clustering complete!")
        self.after(2000, lambda: self.status_label.pack_forget())
        self.run_btn.configure(state="normal", text="Run Clustering")
        self.dendro_btn.configure(state="normal")
        self.is_running = False
        
        # Update Insights
        self.insights_text.configure(state="normal")
        self.insights_text.delete("0.0", "end")
        self.insights_text.insert("0.0", f"Hierarchical Clustering Results:\n\n")
        self.insights_text.insert("end", f"Linkage Method: {linkage_method}\n")
        self.insights_text.insert("end", f"Silhouette Score: {sil_score:.3f}\n\n")
        
        if sil_score > 0.5:
            self.insights_text.insert("end", "Interpretation: The clusters are well-separated and dense.\n")
        elif sil_score > 0.2:
             self.insights_text.insert("end", "Interpretation: The clusters are reasonably separated, but there may be some overlap.\n")
        else:
             self.insights_text.insert("end", "Interpretation: The clusters are overlapping or the data is not well-clustered.\n")
        
        self.insights_text.configure(state="disabled")
        
        self.plot_scatter(X, labels, feature_names, linkage_method, sil_score)

    def _handle_error(self, error_msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()
        self.run_btn.configure(state="normal", text="Run Clustering")
        self.dendro_btn.configure(state="normal")
        self.is_running = False
        tk.messagebox.showerror("Error", f"Clustering failed: {error_msg}")

    def show_dendrogram(self):
        X, _ = self.get_data()
        if X is None: return
        
        # Compute linkage matrix
        if len(X) > 1000:
             if not tk.messagebox.askyesno("Large Dataset", "Dataset is large (>1000 samples). Dendrogram might take a while. Continue?"):
                 return
             # Sample for performance
             sample_idx = np.random.choice(len(X), 1000, replace=False)
             X = X[sample_idx]
        
        self.is_running = True
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.start()
        self.status_label.configure(text="Calculating Dendrogram...")
        self.status_label.pack(fill="x", pady=(0, 10))
        self.run_btn.configure(state="disabled")
        self.dendro_btn.configure(state="disabled")
        
        metric = self.metric_var.get()
        method = self.linkage_var.get()
        
        if method == 'ward' and metric != 'euclidean':
             metric = 'euclidean'
             
        thread = threading.Thread(target=self._run_dendrogram_thread, args=(X, method, metric))
        thread.daemon = True
        thread.start()

    def _run_dendrogram_thread(self, X, method, metric):
        try:
            Z = linkage(X, method=method, metric=metric)
            self.after(0, lambda: self._finish_dendrogram(Z))
        except Exception as e:
            self.after(0, lambda err=str(e): self._handle_error(err))

    def _finish_dendrogram(self, Z):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()
        self.run_btn.configure(state="normal")
        self.dendro_btn.configure(state="normal")
        self.is_running = False
        
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        fig = Figure(figsize=(6, 4.5), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        dendrogram(Z, ax=ax, truncate_mode='level', p=5, color_threshold=0.7*max(Z[:,2]))
        ax.set_title("Hierarchical Dendrogram", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Sample Index / Cluster Size", fontsize=10, fontweight='bold')
        ax.set_ylabel("Distance", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--', axis='y')
        ax.set_facecolor('#FAFAFA')
        
        fig.tight_layout(pad=1.5)
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def plot_scatter(self, X, labels, feature_names, linkage_method, sil_score):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        fig = Figure(figsize=(6, 4.5), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel(feature_names[0], fontsize=10, fontweight='bold')
        ax.set_ylabel(feature_names[1], fontsize=10, fontweight='bold')
        ax.set_title(f'Hierarchical ({linkage_method})\nSilhouette={sil_score:.3f}', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_facecolor('#FAFAFA')
        
        fig.tight_layout(pad=1.5)
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
