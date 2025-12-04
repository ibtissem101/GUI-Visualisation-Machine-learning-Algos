import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.cluster import AgglomerativeClustering
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
        
        # Content Layout
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        
        # Left Panel - Controls
        controls_panel = ctk.CTkFrame(
            content, 
            fg_color="white", 
            corner_radius=8, 
            border_width=1, 
            border_color="#E2E8F0",
            width=280
        )
        controls_panel.pack(side="left", fill="y", padx=(0, 20))
        controls_panel.pack_propagate(False)
        
        # Controls content with padding
        controls_inner = ctk.CTkFrame(controls_panel, fg_color="transparent")
        controls_inner.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(
            controls_inner, 
            text="Parameters", 
            font=("Segoe UI", 16, "bold"), 
            text_color="#1E293B",
            anchor="w"
        ).pack(fill="x", pady=(0, 20))
        
        # Number of Clusters
        ctk.CTkLabel(
            controls_inner, 
            text="Number of Clusters (k)", 
            text_color="#475569",
            font=("Segoe UI", 12),
            anchor="w"
        ).pack(fill="x", pady=(0, 8))
        
        self.k_entry = ctk.CTkEntry(
            controls_inner, 
            placeholder_text="Enter k value",
            height=36,
            font=("Segoe UI", 12)
        )
        self.k_entry.insert(0, "3")
        self.k_entry.pack(fill="x", pady=(0, 20))
        
        # Linkage
        ctk.CTkLabel(
            controls_inner, 
            text="Linkage Method", 
            text_color="#475569",
            font=("Segoe UI", 12),
            anchor="w"
        ).pack(fill="x", pady=(0, 8))
        
        self.linkage_var = ctk.StringVar(value="ward")
        self.linkage_menu = ctk.CTkOptionMenu(
            controls_inner,
            values=["ward", "complete", "average", "single"],
            variable=self.linkage_var,
            fg_color="#2D5BFF",
            button_color="#2D5BFF",
            button_hover_color="#1E40AF",
            dropdown_fg_color="white",
            height=36,
            font=("Segoe UI", 12)
        )
        self.linkage_menu.pack(fill="x", pady=(0, 20))
        
        # Features info
        info_box = ctk.CTkFrame(controls_inner, fg_color="#EFF6FF", corner_radius=6)
        info_box.pack(fill="x", pady=(10, 20))
        
        ctk.CTkLabel(
            info_box, 
            text="ℹ️  Use dendrogram to find\noptimal number of clusters.", 
            text_color="#1E40AF",
            font=("Segoe UI", 11),
            justify="left"
        ).pack(padx=12, pady=12, anchor="w")
        
        # Spacer
        ctk.CTkFrame(controls_inner, fg_color="transparent", height=1).pack(expand=True)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(controls_inner, mode="indeterminate")
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.pack_forget()
        
        self.status_label = ctk.CTkLabel(
            controls_inner,
            text="",
            text_color="#64748B",
            font=("Segoe UI", 11)
        )
        self.status_label.pack(fill="x", pady=(0, 10))
        self.status_label.pack_forget()
        
        # Run Button
        self.run_btn = ctk.CTkButton(
            controls_inner,
            text="Run Clustering",
            command=self.run_clustering,
            font=("Segoe UI", 13, "bold"),
            fg_color="#2D5BFF",
            hover_color="#1E40AF",
            height=44
        )
        self.run_btn.pack(fill="x", pady=(0, 10))
        
        # Dendrogram Button
        self.dendro_btn = ctk.CTkButton(
            controls_inner,
            text="Show Dendrogram",
            command=self.show_dendrogram,
            font=("Segoe UI", 13, "bold"),
            fg_color="#64748B",
            hover_color="#475569",
            height=44
        )
        self.dendro_btn.pack(fill="x", pady=(0, 0))
        
        # Right Panel - Visualization
        self.viz_panel = ctk.CTkFrame(
            content, 
            fg_color="white", 
            corner_radius=8, 
            border_width=1, 
            border_color="#E2E8F0"
        )
        self.viz_panel.pack(side="right", fill="both", expand=True)
        
        # Initial Plot
        self.plot_placeholder()

    def plot_placeholder(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        fig = Figure(figsize=(6, 4.5), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Load data and run clustering\nto see results', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='#64748B', fontsize=14)
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def get_data(self):
        df = self.app.get_dataframe()
        if df is None:
            tk.messagebox.showwarning("No Data", "Please load a dataset first from the Data Loader page.")
            return None, None
            
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            tk.messagebox.showwarning("Insufficient Data", "Dataset needs at least 2 numeric columns.")
            return None, None
            
        return numeric_df.iloc[:, :2].values, numeric_df.columns[:2]

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
        
        # Show progress
        self.is_running = True
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.start()
        self.status_label.configure(text="Running Hierarchical clustering...")
        self.status_label.pack(fill="x", pady=(0, 10))
        self.run_btn.configure(state="disabled", text="Running...")
        self.dendro_btn.configure(state="disabled")
        
        # Run in thread
        thread = threading.Thread(target=self._run_clustering_thread, args=(X, k, linkage_method, feature_names))
        thread.daemon = True
        thread.start()

    def _run_clustering_thread(self, X, k, linkage_method, feature_names):
        try:
            # Sample if large
            if len(X) > 3000:  # Lower threshold for hierarchical
                sample_idx = np.random.choice(len(X), 3000, replace=False)
                X_sample = X[sample_idx]
            else:
                X_sample = X
            
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
            labels = model.fit_predict(X_sample)
            
            self.after(0, lambda: self._finish_clustering(
                X_sample.copy(), 
                labels.copy(), 
                list(feature_names), 
                linkage_method
            ))
        except Exception as e:
            self.after(0, lambda err=str(e): self._handle_error(err))

    def _finish_clustering(self, X, labels, feature_names, linkage_method):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.configure(text="✓ Clustering complete!")
        self.after(2000, lambda: self.status_label.pack_forget())
        self.run_btn.configure(state="normal", text="Run Clustering")
        self.dendro_btn.configure(state="normal")
        self.is_running = False
        
        self.plot_scatter(X, labels, feature_names, linkage_method)

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
                 
        Z = linkage(X, method=self.linkage_var.get())
        
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
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def plot_scatter(self, X, labels, feature_names, linkage_method):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        fig = Figure(figsize=(6, 4.5), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel(feature_names[0], fontsize=10, fontweight='bold')
        ax.set_ylabel(feature_names[1], fontsize=10, fontweight='bold')
        ax.set_title(f'Hierarchical ({linkage_method} linkage)', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_facecolor('#FAFAFA')
        
        fig.tight_layout(pad=1.5)
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
