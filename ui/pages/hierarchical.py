import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
        self.selected_features = []
        
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
            text="Agglomerative clustering builds a hierarchy of clusters by progressively merging the closest clusters.",
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
        
        params_scroll = ctk.CTkScrollableFrame(self.params_frame, fg_color="transparent")
        params_scroll.pack(fill="both", expand=True, padx=20, pady=20)
        
        controls_inner = ctk.CTkFrame(params_scroll, fg_color="transparent")
        controls_inner.pack(fill="both", expand=True, padx=20, pady=0)
        
        ctk.CTkLabel(
            controls_inner, 
            text="Configuration", 
            font=("Segoe UI", 18, "bold"), 
            text_color="#1E293B",
            anchor="w"
        ).pack(fill="x", pady=(0, 20))
        
        grid_frame = ctk.CTkFrame(controls_inner, fg_color="transparent")
        grid_frame.pack(fill="x", pady=(0, 20))
        grid_frame.grid_columnconfigure(0, weight=1)
        grid_frame.grid_columnconfigure(1, weight=1)
        
        # Number of Clusters
        ctk.CTkLabel(grid_frame, text="Number of Clusters", text_color="#475569", font=("Segoe UI", 12), anchor="w").grid(row=0, column=0, padx=10, pady=(0, 5), sticky="w")
        self.n_clusters_entry = ctk.CTkEntry(grid_frame, placeholder_text="Enter value", height=36, font=("Segoe UI", 12))
        self.n_clusters_entry.insert(0, "3")
        self.n_clusters_entry.grid(row=1, column=0, padx=10, pady=(0, 20), sticky="ew")
        
        # Linkage Method
        ctk.CTkLabel(grid_frame, text="Linkage Method", text_color="#475569", font=("Segoe UI", 12), anchor="w").grid(row=0, column=1, padx=10, pady=(0, 5), sticky="w")
        self.linkage_var = ctk.StringVar(value="ward")
        self.linkage_menu = ctk.CTkOptionMenu(
            grid_frame,
            values=["ward", "complete", "average", "single"],
            variable=self.linkage_var,
            height=36,
            font=("Segoe UI", 12),
            fg_color="#F1F5F9",
            text_color="#1E293B"
        )
        self.linkage_menu.grid(row=1, column=1, padx=10, pady=(0, 20), sticky="ew")

        # Distance Metric
        ctk.CTkLabel(grid_frame, text="Distance Metric", text_color="#475569", font=("Segoe UI", 12), anchor="w").grid(row=2, column=0, padx=10, pady=(0, 5), sticky="w")
        self.metric_var = ctk.StringVar(value="euclidean")
        self.metric_menu = ctk.CTkOptionMenu(
            grid_frame,
            values=["euclidean", "manhattan", "cosine"],
            variable=self.metric_var,
            height=36,
            font=("Segoe UI", 12),
            fg_color="#F1F5F9",
            text_color="#1E293B"
        )
        self.metric_menu.grid(row=3, column=0, padx=10, pady=(0, 20), sticky="ew")

        # Feature Selection
        ctk.CTkLabel(
            controls_inner, 
            text="Feature Selection", 
            font=("Segoe UI", 16, "bold"), 
            text_color="#1E293B",
            anchor="w"
        ).pack(fill="x", pady=(10, 5))
        
        ctk.CTkLabel(
            controls_inner, 
            text="Select multiple features for clustering. Visualization uses PCA if more than 2 features selected.",
            font=("Segoe UI", 11),
            text_color="#64748B",
            anchor="w"
        ).pack(fill="x", pady=(0, 10))
        
        self.features_frame = ctk.CTkFrame(controls_inner, fg_color="#F9FAFB", corner_radius=8)
        self.features_frame.pack(fill="x", pady=(0, 10))
        
        self.feature_vars = {}
        self.feature_checkboxes_frame = ctk.CTkFrame(self.features_frame, fg_color="transparent")
        self.feature_checkboxes_frame.pack(fill="x", padx=15, pady=15)
        
        btn_row = ctk.CTkFrame(controls_inner, fg_color="transparent")
        btn_row.pack(fill="x", pady=(0, 10))
        
        ctk.CTkButton(
            btn_row, text="Select All", width=100, height=28,
            font=("Segoe UI", 11), fg_color="#64748B", hover_color="#475569",
            command=self.select_all_features
        ).pack(side="left", padx=(0, 10))
        
        ctk.CTkButton(
            btn_row, text="Deselect All", width=100, height=28,
            font=("Segoe UI", 11), fg_color="#64748B", hover_color="#475569",
            command=self.deselect_all_features
        ).pack(side="left")
        
        self.features_count_label = ctk.CTkLabel(
            btn_row, text="0 features selected", 
            font=("Segoe UI", 11), text_color="#64748B"
        )
        self.features_count_label.pack(side="right")
        
        self.bind("<Visibility>", self.update_feature_options)

        self.progress_bar = ctk.CTkProgressBar(controls_inner, mode="indeterminate")
        
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
        
        viz_controls = ctk.CTkFrame(self.viz_frame, fg_color="transparent")
        viz_controls.pack(fill="x", padx=20, pady=(20, 10))
        
        self.dendro_btn = ctk.CTkButton(
            viz_controls,
            text="🌳 Show Dendrogram",
            command=self.show_dendrogram,
            font=("Segoe UI", 12, "bold"),
            fg_color="#64748B",
            hover_color="#475569",
            height=36,
            width=180
        )
        self.dendro_btn.pack(side="left")
        
        self.pca_label = ctk.CTkLabel(
            viz_controls, text="", font=("Segoe UI", 11), text_color="#64748B"
        )
        self.pca_label.pack(side="right")
        
        self.plot_container = ctk.CTkFrame(self.viz_frame, fg_color="transparent")
        self.plot_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
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
        for widget in self.feature_checkboxes_frame.winfo_children():
            widget.destroy()
        self.feature_vars.clear()
        
        df = self.app.get_dataframe()
        if df is None:
            ctk.CTkLabel(
                self.feature_checkboxes_frame,
                text="⚠️ Please load a dataset first from the Data Loader page",
                font=("Segoe UI", 12),
                text_color="#64748B"
            ).pack(pady=10)
            self.features_count_label.configure(text="No data loaded")
            return
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            ctk.CTkLabel(
                self.feature_checkboxes_frame,
                text="⚠️ No numeric columns found in dataset",
                font=("Segoe UI", 12),
                text_color="#64748B"
            ).pack(pady=10)
            self.features_count_label.configure(text="No numeric columns")
            return
        
        for i, col in enumerate(numeric_cols):
            var = ctk.BooleanVar(value=i < 2)
            self.feature_vars[col] = var
            
            cb = ctk.CTkCheckBox(
                self.feature_checkboxes_frame,
                text=col[:20] + "..." if len(col) > 20 else col,
                variable=var,
                font=("Segoe UI", 11),
                fg_color="#3B82F6",
                command=self.update_feature_count
            )
            row = i // 3
            col_idx = i % 3
            cb.grid(row=row, column=col_idx, padx=10, pady=5, sticky="w")
        
        self.update_feature_count()
    
    def update_feature_count(self):
        count = sum(1 for var in self.feature_vars.values() if var.get())
        self.features_count_label.configure(text=f"{count} features selected")
    
    def select_all_features(self):
        for var in self.feature_vars.values():
            var.set(True)
        self.update_feature_count()
    
    def deselect_all_features(self):
        for var in self.feature_vars.values():
            var.set(False)
        self.update_feature_count()
    
    def get_selected_features(self):
        return [col for col, var in self.feature_vars.items() if var.get()]

    def run_clustering(self):
        if self.is_running:
            return
            
        df = self.app.get_dataframe()
        if df is None:
            tk.messagebox.showwarning("No Data", "Please load a dataset first from the Data Loader page.")
            return
        
        selected_features = self.get_selected_features()
        if len(selected_features) < 2:
            tk.messagebox.showwarning("Insufficient Features", "Please select at least 2 features for clustering.")
            return
            
        try:
            n_clusters = int(self.n_clusters_entry.get())
            linkage_method = self.linkage_var.get()
            metric = self.metric_var.get()
            
            # Ward only works with euclidean
            if linkage_method == "ward" and metric != "euclidean":
                metric = "euclidean"
            
            if n_clusters < 2:
                tk.messagebox.showerror("Invalid Input", "Number of clusters must be at least 2.")
                return
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Please enter valid integers for parameters.")
            return
        
        self.is_running = True
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.start()
        self.status_label.configure(text="Running Hierarchical clustering...")
        self.status_label.pack(fill="x", pady=(0, 10))
        self.run_btn.configure(state="disabled", text="Running...")
        self.dendro_btn.configure(state="disabled")
        
        thread = threading.Thread(target=self._run_clustering_thread, args=(df, n_clusters, linkage_method, metric, selected_features))
        thread.daemon = True
        thread.start()

    def _run_clustering_thread(self, df, n_clusters, linkage_method, metric, selected_features):
        try:
            X = df[selected_features].values
            X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
            
            if len(X) > 5000:
                sample_idx = np.random.choice(len(X), 5000, replace=False)
                X_sample = X[sample_idx]
            else:
                X_sample = X
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sample)
            
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage_method,
                metric=metric if linkage_method != "ward" else "euclidean"
            )
            labels = model.fit_predict(X_scaled)
            
            if len(set(labels)) > 1:
                sil_score = silhouette_score(X_scaled, labels)
            else:
                sil_score = -1.0
            
            # PCA for visualization
            if len(selected_features) > 2:
                pca = PCA(n_components=2)
                X_viz = pca.fit_transform(X_scaled)
                explained_var = sum(pca.explained_variance_ratio_) * 100
                viz_labels = [f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", 
                             f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"]
            else:
                X_viz = X_sample
                explained_var = None
                viz_labels = selected_features[:2]
            
            self.after(0, lambda: self._finish_clustering(
                X_viz.copy(), labels.copy(), list(viz_labels), 
                linkage_method, sil_score, len(selected_features), explained_var
            ))
        except Exception as e:
            self.after(0, lambda err=str(e): self._handle_error(err))

    def _finish_clustering(self, X, labels, viz_labels, linkage_method, sil_score, n_features, explained_var):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.configure(text="✓ Clustering complete!")
        self.after(2000, lambda: self.status_label.pack_forget())
        self.run_btn.configure(state="normal", text="Run Clustering")
        self.dendro_btn.configure(state="normal")
        self.is_running = False
        
        if explained_var:
            self.pca_label.configure(text=f"📐 Using PCA: {explained_var:.1f}% variance explained from {n_features} features")
        else:
            self.pca_label.configure(text=f"📐 Using {n_features} features directly")
        
        self.insights_text.configure(state="normal")
        self.insights_text.delete("0.0", "end")
        self.insights_text.insert("0.0", f"Hierarchical Clustering Results:\n\n")
        self.insights_text.insert("end", f"Linkage Method: {linkage_method}\n")
        self.insights_text.insert("end", f"Features Used: {n_features}\n")
        self.insights_text.insert("end", f"Silhouette Score: {sil_score:.3f}\n\n")
        
        if explained_var:
            self.insights_text.insert("end", f"PCA Visualization: {explained_var:.1f}% variance explained\n\n")
        
        if sil_score > 0.5:
            self.insights_text.insert("end", "Interpretation: The clusters are well-separated and dense.\n")
        elif sil_score > 0.2:
             self.insights_text.insert("end", "Interpretation: The clusters are reasonably separated.\n")
        else:
             self.insights_text.insert("end", "Interpretation: The clusters are overlapping.\n")
        
        self.insights_text.configure(state="disabled")
        
        self.plot_scatter(X, labels, viz_labels, linkage_method, sil_score)
        
        self.view_var.set("Visualization")
        self.switch_view("Visualization")

    def _handle_error(self, error_msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()
        self.run_btn.configure(state="normal", text="Run Clustering")
        self.dendro_btn.configure(state="normal")
        self.is_running = False
        tk.messagebox.showerror("Error", f"Clustering failed: {error_msg}")

    def show_dendrogram(self):
        if self.is_running: return
        
        df = self.app.get_dataframe()
        if df is None:
            tk.messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        selected_features = self.get_selected_features()
        if len(selected_features) < 2:
            tk.messagebox.showwarning("Insufficient Features", "Please select at least 2 features.")
            return
            
        self.is_running = True
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.start()
        self.status_label.configure(text="Generating Dendrogram...")
        self.status_label.pack(fill="x", pady=(0, 10))
        self.run_btn.configure(state="disabled")
        self.dendro_btn.configure(state="disabled")
        
        thread = threading.Thread(target=self._run_dendrogram_thread, args=(df, selected_features))
        thread.daemon = True
        thread.start()

    def _run_dendrogram_thread(self, df, selected_features):
        try:
            X = df[selected_features].values
            X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
            
            # Limit samples for dendrogram
            if len(X) > 1000:
                idx = np.random.choice(len(X), 1000, replace=False)
                X = X[idx]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            linkage_method = self.linkage_var.get()
            Z = linkage(X_scaled, method=linkage_method)
            
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
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.view_var.set("Visualization")
        self.switch_view("Visualization")

    def plot_scatter(self, X, labels, viz_labels, linkage_method, sil_score):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        fig = Figure(figsize=(6, 4.5), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel(viz_labels[0], fontsize=10, fontweight='bold')
        ax.set_ylabel(viz_labels[1], fontsize=10, fontweight='bold')
        ax.set_title(f'Hierarchical ({linkage_method})\nSilhouette={sil_score:.3f}', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_facecolor('#FAFAFA')
        
        fig.tight_layout(pad=1.5)
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
