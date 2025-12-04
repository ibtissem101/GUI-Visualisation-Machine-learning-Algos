import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import threading

class DBSCANPage(ctk.CTkFrame):
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
            text="DBSCAN Clustering",
            font=("Segoe UI", 26, "bold"),
            text_color="#0F172A",
            anchor="w"
        )
        header.pack(padx=30, pady=(30, 5), anchor="w")
        
        subtitle = ctk.CTkLabel(
            self,
            text="Density-Based Spatial Clustering of Applications with Noise - finds arbitrarily shaped clusters.",
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
        
        # Epsilon
        ctk.CTkLabel(grid_frame, text="Epsilon (eps)", text_color="#475569", font=("Segoe UI", 12), anchor="w").grid(row=0, column=0, padx=10, pady=(0, 5), sticky="w")
        self.eps_entry = ctk.CTkEntry(grid_frame, placeholder_text="Max distance", height=36, font=("Segoe UI", 12))
        self.eps_entry.insert(0, "0.5")
        self.eps_entry.grid(row=1, column=0, padx=10, pady=(0, 20), sticky="ew")
        
        # Min Samples
        ctk.CTkLabel(grid_frame, text="Min Samples", text_color="#475569", font=("Segoe UI", 12), anchor="w").grid(row=0, column=1, padx=10, pady=(0, 5), sticky="w")
        self.min_samples_entry = ctk.CTkEntry(grid_frame, placeholder_text="Min points in cluster", height=36, font=("Segoe UI", 12))
        self.min_samples_entry.insert(0, "5")
        self.min_samples_entry.grid(row=1, column=1, padx=10, pady=(0, 20), sticky="ew")

        # Metric
        ctk.CTkLabel(grid_frame, text="Metric", text_color="#475569", font=("Segoe UI", 12), anchor="w").grid(row=2, column=0, padx=10, pady=(0, 5), sticky="w")
        self.metric_var = ctk.StringVar(value="euclidean")
        self.metric_menu = ctk.CTkOptionMenu(
            grid_frame,
            values=["euclidean", "l1", "l2", "manhattan", "cosine"],
            variable=self.metric_var,
            height=36,
            font=("Segoe UI", 12),
            fg_color="#F1F5F9",
            text_color="#1E293B",
            button_color="#CBD5E1",
            button_hover_color="#94A3B8"
        )
        self.metric_menu.grid(row=3, column=0, padx=10, pady=(0, 20), sticky="ew")

        # Algorithm
        ctk.CTkLabel(grid_frame, text="Algorithm", text_color="#475569", font=("Segoe UI", 12), anchor="w").grid(row=2, column=1, padx=10, pady=(0, 5), sticky="w")
        self.algo_var = ctk.StringVar(value="auto")
        self.algo_menu = ctk.CTkOptionMenu(
            grid_frame,
            values=["auto", "ball_tree", "kd_tree", "brute"],
            variable=self.algo_var,
            height=36,
            font=("Segoe UI", 12),
            fg_color="#F1F5F9",
            text_color="#1E293B",
            button_color="#CBD5E1",
            button_hover_color="#94A3B8"
        )
        self.algo_menu.grid(row=3, column=1, padx=10, pady=(0, 20), sticky="ew")
        
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

        # Features info
        info_box = ctk.CTkFrame(controls_inner, fg_color="#FEF3C7", corner_radius=6)
        info_box.pack(fill="x", pady=(10, 20))
        
        ctk.CTkLabel(
            info_box, 
            text="⚠️  Points labeled -1 are\nconsidered noise/outliers.", 
            text_color="#92400E",
            font=("Segoe UI", 11),
            justify="left"
        ).pack(padx=12, pady=12, anchor="w")
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(controls_inner, mode="indeterminate")
        
        # Buttons
        btn_frame = ctk.CTkFrame(controls_inner, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(20, 0))
        
        self.run_btn = ctk.CTkButton(
            btn_frame,
            text="Run Clustering",
            command=self.run_dbscan,
            font=("Segoe UI", 13, "bold"),
            height=44,
            fg_color="#2D5BFF",
            hover_color="#1E40AF"
        )
        self.run_btn.pack(fill="x", expand=True)
        
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
        
        # Plot container
        self.plot_container = ctk.CTkFrame(self.viz_frame, fg_color="transparent")
        self.plot_container.pack(fill="both", expand=True, padx=10, pady=10)
        
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
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

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

    def run_dbscan(self):
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
            eps = float(self.eps_entry.get())
            min_samples = int(self.min_samples_entry.get())
            metric = self.metric_var.get()
            algorithm = self.algo_var.get()
            
            if eps <= 0:
                tk.messagebox.showerror("Invalid Input", "Epsilon must be greater than 0.")
                return
            if min_samples < 1:
                tk.messagebox.showerror("Invalid Input", "Min samples must be at least 1.")
                return
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Please check your parameters.")
            return
            
        x_col = self.x_axis_var.get()
        y_col = self.y_axis_var.get()
        
        if not x_col or not y_col:
            self.update_feature_options()
            x_col = self.x_axis_var.get()
            y_col = self.y_axis_var.get()
        
        # Show progress
        self.is_running = True
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.start()
        self.status_label.configure(text="Running DBSCAN clustering...")
        self.status_label.pack(fill="x", pady=(0, 10))
        self.run_btn.configure(state="disabled", text="Running...")
        
        # Run in thread
        thread = threading.Thread(target=self._run_dbscan_thread, args=(numeric_df, eps, min_samples, metric, algorithm, x_col, y_col))
        thread.daemon = True
        thread.start()

    def _run_dbscan_thread(self, numeric_df, eps, min_samples, metric, algorithm, x_col, y_col):
        try:
            X = numeric_df[[x_col, y_col]].values
            feature_names = [x_col, y_col]
            
            # Sample if large
            if len(X) > 5000:
                sample_idx = np.random.choice(len(X), 5000, replace=False)
                X_sample = X[sample_idx]
            else:
                X_sample = X
            
            # DBSCAN with parallel processing
            db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, n_jobs=-1)
            labels = db.fit_predict(X_sample)
            
            # Calculate Silhouette Score (ignoring noise if possible, or including it)
            # Usually silhouette is not defined for noise (-1), but we can include it or exclude it.
            # Let's exclude noise for silhouette calculation if possible, or just calculate for all.
            # If all points are noise, silhouette is not defined.
            unique_labels = set(labels)
            if len(unique_labels) > 1:
                # Filter out noise for silhouette score if there are clusters
                if -1 in unique_labels and len(unique_labels) > 2:
                    mask = labels != -1
                    if np.sum(mask) > 2:
                        sil_score = silhouette_score(X_sample[mask], labels[mask])
                    else:
                        sil_score = -1.0
                elif -1 not in unique_labels:
                    sil_score = silhouette_score(X_sample, labels)
                else:
                    sil_score = -1.0 # Only noise and one cluster or just noise
            else:
                sil_score = -1.0
            
            self.after(0, lambda: self._finish_dbscan(
                X_sample.copy(), 
                labels.copy(), 
                list(feature_names),
                sil_score
            ))
        except Exception as e:
            self.after(0, lambda err=str(e): self._handle_error(err))

    def _finish_dbscan(self, X, labels, feature_names, sil_score):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.configure(text="✓ Clustering complete!")
        self.after(2000, lambda: self.status_label.pack_forget())
        self.run_btn.configure(state="normal", text="Run Clustering")
        self.is_running = False
        
        # Update Insights
        self.insights_text.configure(state="normal")
        self.insights_text.delete("0.0", "end")
        self.insights_text.insert("0.0", f"DBSCAN Results:\n\n")
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        self.insights_text.insert("end", f"Estimated Number of Clusters: {n_clusters}\n")
        self.insights_text.insert("end", f"Noise Points: {n_noise}\n")
        self.insights_text.insert("end", f"Silhouette Score (excluding noise): {sil_score:.3f}\n\n")
        
        if n_clusters == 0:
             self.insights_text.insert("end", "Interpretation: No clusters found. Try adjusting Epsilon or Min Samples.\n")
        elif sil_score > 0.5:
            self.insights_text.insert("end", "Interpretation: The clusters are well-separated and dense.\n")
        elif sil_score > 0.2:
             self.insights_text.insert("end", "Interpretation: The clusters are reasonably separated, but there may be some overlap.\n")
        else:
             self.insights_text.insert("end", "Interpretation: The clusters are overlapping or the data is not well-clustered.\n")
        
        self.insights_text.configure(state="disabled")
        
        self.plot_results(X, labels, feature_names, sil_score)
        
        # Auto-switch to Visualization view to show results
        self.view_var.set("Visualization")
        self.switch_view("Visualization")

    def _handle_error(self, error_msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()
        self.run_btn.configure(state="normal", text="Run Clustering")
        self.is_running = False
        tk.messagebox.showerror("Error", f"Clustering failed: {error_msg}")

    def plot_results(self, X, labels, feature_names, sil_score):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        fig = Figure(figsize=(6, 4.5), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        # Plot noise points
        if -1 in unique_labels:
            noise_mask = labels == -1
            ax.scatter(X[noise_mask, 0], X[noise_mask, 1], c='gray', marker='x', s=20, alpha=0.5, label='Noise')
        
        # Plot cluster points
        cluster_mask = labels != -1
        if cluster_mask.any():
            scatter = ax.scatter(X[cluster_mask, 0], X[cluster_mask, 1], c=labels[cluster_mask], 
                               cmap='viridis', alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel(feature_names[0], fontsize=10, fontweight='bold')
        ax.set_ylabel(feature_names[1], fontsize=10, fontweight='bold')
        ax.set_title(f'DBSCAN (Clusters: {n_clusters}, Noise: {n_noise})\nSilhouette={sil_score:.3f}', fontsize=12, fontweight='bold', pad=10)
        ax.legend(frameon=True, shadow=True, fontsize=9)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_facecolor('#FAFAFA')
        
        fig.tight_layout(pad=1.5)
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
