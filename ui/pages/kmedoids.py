import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score
import pandas as pd
import numpy as np
import threading

class KMedoidsPage(ctk.CTkFrame):
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
            text="K-Medoids Clustering",
            font=("Segoe UI", 26, "bold"),
            text_color="#0F172A",
            anchor="w"
        )
        header.pack(padx=30, pady=(30, 5), anchor="w")
        
        subtitle = ctk.CTkLabel(
            self,
            text="Similar to K-Means but uses actual data points as centers (medoids), making it more robust to outliers.",
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
        
        # Max Iterations
        ctk.CTkLabel(grid_frame, text="Max Iterations", text_color="#475569", font=("Segoe UI", 12), anchor="w").grid(row=0, column=1, padx=10, pady=(0, 5), sticky="w")
        self.max_iter_entry = ctk.CTkEntry(grid_frame, placeholder_text="Default: 300", height=36, font=("Segoe UI", 12))
        self.max_iter_entry.insert(0, "300")
        self.max_iter_entry.grid(row=1, column=1, padx=10, pady=(0, 20), sticky="ew")

        # Initialization
        ctk.CTkLabel(grid_frame, text="Initialization", text_color="#475569", font=("Segoe UI", 12), anchor="w").grid(row=2, column=0, padx=10, pady=(0, 5), sticky="w")
        self.init_var = ctk.StringVar(value="k-means++")
        self.init_menu = ctk.CTkOptionMenu(
            grid_frame,
            values=["k-means++", "random"],
            variable=self.init_var,
            height=36,
            font=("Segoe UI", 12),
            fg_color="#F1F5F9",
            text_color="#1E293B",
            button_color="#CBD5E1",
            button_hover_color="#94A3B8"
        )
        self.init_menu.grid(row=3, column=0, padx=10, pady=(0, 20), sticky="ew")
        
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
            command=self.run_kmedoids,
            font=("Segoe UI", 13, "bold"),
            height=44,
            fg_color="#2D5BFF",
            hover_color="#1E40AF"
        )
        self.run_btn.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.elbow_btn = ctk.CTkButton(
            btn_frame,
            text="Show Elbow Method",
            command=self.show_elbow,
            font=("Segoe UI", 13, "bold"),
            fg_color="#64748B",
            hover_color="#475569",
            height=44
        )
        self.elbow_btn.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
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

    def run_kmedoids(self):
        if self.is_running:
            return
            
        df = self.app.get_dataframe()
        if df is None:
            tk.messagebox.showwarning("No Data", "Please load a dataset first from the Data Loader page.")
            return
            
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            tk.messagebox.showwarning("Insufficient Data", "Dataset needs at least 2 numeric columns.")
            return
            
        x_col = self.x_axis_var.get()
        y_col = self.y_axis_var.get()
        
        if not x_col or not y_col:
            self.update_feature_options()
            x_col = self.x_axis_var.get()
            y_col = self.y_axis_var.get()
            
        try:
            k = int(self.k_entry.get())
            max_iter = int(self.max_iter_entry.get())
            init_method = self.init_var.get()
            
            if k < 2:
                tk.messagebox.showerror("Invalid Input", "Number of clusters must be at least 2.")
                return
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Number of clusters must be an integer.")
            return
        
        # Show progress
        self.is_running = True
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.start()
        self.status_label.configure(text="Running K-Medoids clustering...")
        self.status_label.pack(fill="x", pady=(0, 10))
        self.run_btn.configure(state="disabled", text="Running...")
        self.elbow_btn.configure(state="disabled")
        
        # Run in thread
        thread = threading.Thread(target=self._run_kmedoids_thread, args=(numeric_df, k, max_iter, init_method, x_col, y_col))
        thread.daemon = True
        thread.start()

    def _run_kmedoids_thread(self, numeric_df, k, max_iter, init_method, x_col, y_col):
        try:
            X = numeric_df[[x_col, y_col]].values
            feature_names = [x_col, y_col]
            
            # Sample if large
            if len(X) > 5000:
                sample_idx = np.random.choice(len(X), 5000, replace=False)
                X_sample = X[sample_idx]
            else:
                X_sample = X
            
            # Optimize KMeans for speed
            kmeans = KMeans(
                n_clusters=k, 
                n_init=10, 
                max_iter=max_iter,
                init=init_method,
                algorithm='elkan', 
                random_state=42
            )
            kmeans.fit(X_sample)
            
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_sample)
            medoids = X_sample[closest]
            labels = kmeans.predict(X_sample)
            
            # Calculate Silhouette Score
            if len(set(labels)) > 1:
                sil_score = silhouette_score(X_sample, labels)
            else:
                sil_score = -1.0
            
            self.after(0, lambda: self._finish_kmedoids(
                X_sample.copy(), 
                labels.copy(), 
                medoids.copy(), 
                list(feature_names),
                sil_score
            ))
        except Exception as e:
            self.after(0, lambda err=str(e): self._handle_error(err))

    def _finish_kmedoids(self, X, labels, medoids, feature_names, sil_score):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.configure(text="✓ Clustering complete!")
        self.after(2000, lambda: self.status_label.pack_forget())
        self.run_btn.configure(state="normal", text="Run Clustering")
        self.elbow_btn.configure(state="normal")
        self.is_running = False
        
        # Update Insights
        self.insights_text.configure(state="normal")
        self.insights_text.delete("0.0", "end")
        self.insights_text.insert("0.0", f"K-Medoids Results:\n\n")
        self.insights_text.insert("end", f"Number of Clusters (k): {len(medoids)}\n")
        self.insights_text.insert("end", f"Silhouette Score: {sil_score:.3f}\n\n")
        
        if sil_score > 0.5:
            self.insights_text.insert("end", "Interpretation: The clusters are well-separated and dense.\n")
        elif sil_score > 0.2:
             self.insights_text.insert("end", "Interpretation: The clusters are reasonably separated, but there may be some overlap.\n")
        else:
             self.insights_text.insert("end", "Interpretation: The clusters are overlapping or the data is not well-clustered.\n")
        
        self.insights_text.configure(state="disabled")
        
        self.plot_results(X, labels, medoids, feature_names, sil_score)

    def _handle_error(self, error_msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()
        self.run_btn.configure(state="normal", text="Run Clustering")
        self.elbow_btn.configure(state="normal")
        self.is_running = False
        tk.messagebox.showerror("Error", f"Clustering failed: {error_msg}")

    def plot_results(self, X, labels, medoids, feature_names, sil_score):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        fig = Figure(figsize=(6, 4.5), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
        ax.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='*', s=300, edgecolors='black', linewidth=1.5, label='Medoids', zorder=5)
        
        ax.set_xlabel(feature_names[0], fontsize=10, fontweight='bold')
        ax.set_ylabel(feature_names[1], fontsize=10, fontweight='bold')
        ax.set_title(f'K-Medoids (k={len(medoids)})\nSilhouette={sil_score:.3f}', fontsize=12, fontweight='bold', pad=10)
        ax.legend(frameon=True, shadow=True, fontsize=9)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_facecolor('#FAFAFA')
        
        fig.tight_layout(pad=1.5)
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def show_elbow(self):
        if self.is_running: return
        
        df = self.app.get_dataframe()
        if df is None:
            tk.messagebox.showwarning("No Data", "Please load a dataset first.")
            return
            
        numeric_df = df.select_dtypes(include=[np.number])
        x_col = self.x_axis_var.get()
        y_col = self.y_axis_var.get()
        
        if not x_col or not y_col:
            self.update_feature_options()
            x_col = self.x_axis_var.get()
            y_col = self.y_axis_var.get()
            
        self.is_running = True
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.start()
        self.status_label.configure(text="Calculating Elbow Method...")
        self.status_label.pack(fill="x", pady=(0, 10))
        self.run_btn.configure(state="disabled")
        self.elbow_btn.configure(state="disabled")
        
        thread = threading.Thread(target=self._run_elbow_thread, args=(numeric_df, x_col, y_col))
        thread.daemon = True
        thread.start()

    def _run_elbow_thread(self, numeric_df, x_col, y_col):
        try:
            X = numeric_df[[x_col, y_col]].values
            if len(X) > 3000:
                idx = np.random.choice(len(X), 3000, replace=False)
                X = X[idx]
                
            inertias = []
            K = range(1, 11)
            
            for k in K:
                kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
                
            self.after(0, lambda: self._finish_elbow(K, inertias))
        except Exception as e:
            self.after(0, lambda err=str(e): self._handle_error(err))

    def _finish_elbow(self, K, inertias):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()
        self.run_btn.configure(state="normal")
        self.elbow_btn.configure(state="normal")
        self.is_running = False
        
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        fig = Figure(figsize=(6, 4.5), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        ax.plot(K, inertias, 'bx-')
        ax.set_xlabel('k (Number of clusters)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method For Optimal k')
        ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def _handle_error(self, error_msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()
        self.run_btn.configure(state="normal", text="Run Clustering")
        self.is_running = False
        tk.messagebox.showerror("Error", f"Clustering failed: {error_msg}")

    def plot_results(self, X, labels, medoids, feature_names):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        fig = Figure(figsize=(6, 4.5), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
        ax.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='*', s=300, edgecolors='black', linewidth=1.5, label='Medoids', zorder=5)
        
        ax.set_xlabel(feature_names[0], fontsize=10, fontweight='bold')
        ax.set_ylabel(feature_names[1], fontsize=10, fontweight='bold')
        ax.set_title(f'K-Medoids (k={len(medoids)})', fontsize=12, fontweight='bold', pad=10)
        ax.legend(frameon=True, shadow=True, fontsize=9)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_facecolor('#FAFAFA')
        
        fig.tight_layout(pad=1.5)
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
