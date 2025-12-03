import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import threading

class KMeansPage(ctk.CTkFrame):
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
            text="K-Means Clustering",
            font=("Segoe UI", 26, "bold"),
            text_color="#0F172A",
            anchor="w"
        )
        header.pack(padx=30, pady=(30, 5), anchor="w")
        
        subtitle = ctk.CTkLabel(
            self,
            text="Partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean.",
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
        
        # Max Iterations
        ctk.CTkLabel(
            controls_inner, 
            text="Max Iterations", 
            text_color="#475569",
            font=("Segoe UI", 12),
            anchor="w"
        ).pack(fill="x", pady=(0, 8))
        
        self.max_iter_entry = ctk.CTkEntry(
            controls_inner, 
            placeholder_text="Default: 300",
            height=36,
            font=("Segoe UI", 12)
        )
        self.max_iter_entry.insert(0, "300")
        self.max_iter_entry.pack(fill="x", pady=(0, 20))
        
        # Features info
        info_box = ctk.CTkFrame(controls_inner, fg_color="#EFF6FF", corner_radius=6)
        info_box.pack(fill="x", pady=(10, 20))
        
        ctk.CTkLabel(
            info_box, 
            text="ℹ️  Will use first 2 numeric\ncolumns for visualization.", 
            text_color="#1E40AF",
            font=("Segoe UI", 11),
            justify="left"
        ).pack(padx=12, pady=12, anchor="w")
        
        # Spacer
        ctk.CTkFrame(controls_inner, fg_color="transparent", height=1).pack(expand=True)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(controls_inner, mode="indeterminate")
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.pack_forget()  # Hide initially
        
        self.status_label = ctk.CTkLabel(
            controls_inner,
            text="",
            text_color="#64748B",
            font=("Segoe UI", 11)
        )
        self.status_label.pack(fill="x", pady=(0, 10))
        self.status_label.pack_forget()  # Hide initially
        
        # Run Button
        self.run_btn = ctk.CTkButton(
            controls_inner,
            text="Run Clustering",
            command=self.run_kmeans,
            font=("Segoe UI", 13, "bold"),
            fg_color="#2D5BFF",
            hover_color="#1E40AF",
            height=44
        )
        self.run_btn.pack(fill="x", pady=(0, 0))
        
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
            
        fig = Figure(figsize=(8, 6), dpi=100, facecolor='white')
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
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)

    def run_kmeans(self):
        if self.is_running:
            return
            
        df = self.app.get_dataframe()
        if df is None:
            tk.messagebox.showwarning("No Data", "Please load a dataset first from the Data Loader page.")
            return
            
        # Get numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            tk.messagebox.showwarning("Insufficient Data", "Dataset needs at least 2 numeric columns.")
            return
            
        try:
            k = int(self.k_entry.get())
            max_iter = int(self.max_iter_entry.get())
            
            if k < 2:
                tk.messagebox.showerror("Invalid Input", "Number of clusters must be at least 2.")
                return
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Please enter valid integers for parameters.")
            return
        
        # Show progress
        self.is_running = True
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.start()
        self.status_label.configure(text="Running K-Means clustering...")
        self.status_label.pack(fill="x", pady=(0, 10))
        self.run_btn.configure(state="disabled", text="Running...")
        
        # Run in thread
        thread = threading.Thread(target=self._run_kmeans_thread, args=(numeric_df, k, max_iter))
        thread.daemon = True
        thread.start()

    def _run_kmeans_thread(self, numeric_df, k, max_iter):
        try:
            # Only do computation in thread
            X = numeric_df.iloc[:, :2].values
            feature_names = numeric_df.columns[:2]
            
            kmeans = KMeans(n_clusters=k, n_init=10, max_iter=max_iter, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # Schedule UI update on main thread with results
            self.after(0, lambda: self._finish_kmeans(
                X.copy(), 
                labels.copy(), 
                kmeans.cluster_centers_.copy(), 
                list(feature_names), 
                float(kmeans.inertia_)
            ))
        except Exception as e:
            self.after(0, lambda err=str(e): self._handle_error(err))

    def _finish_kmeans(self, X, labels, centers, feature_names, inertia):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.configure(text="✓ Clustering complete!")
        self.after(2000, lambda: self.status_label.pack_forget())
        self.run_btn.configure(state="normal", text="Run Clustering")
        self.is_running = False
        
        self.plot_results(X, labels, centers, feature_names, inertia)

    def _handle_error(self, error_msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()
        self.run_btn.configure(state="normal", text="Run Clustering")
        self.is_running = False
        tk.messagebox.showerror("Error", f"Clustering failed: {error_msg}")

    def plot_results(self, X, labels, centers, feature_names, inertia):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        fig = Figure(figsize=(8, 6), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, edgecolors='black', linewidth=1.5, label='Centroids', zorder=5)
        
        ax.set_xlabel(feature_names[0], fontsize=11, fontweight='bold')
        ax.set_ylabel(feature_names[1], fontsize=11, fontweight='bold')
        ax.set_title(f'K-Means Clustering (k={len(centers)}, Inertia={inertia:.2f})', fontsize=13, fontweight='bold', pad=15)
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_facecolor('#FAFAFA')
        
        fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
