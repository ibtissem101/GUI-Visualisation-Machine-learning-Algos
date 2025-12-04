import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import threading

class ComparisonPage(ctk.CTkFrame):
    def __init__(self, master, app_instance):
        super().__init__(master, fg_color="#F5F5F5", corner_radius=0)
        self.app = app_instance
        self.figure = None
        self.canvas = None
        self.is_running = False
        self.results = []
        self.X_viz = None
        self.viz_labels = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header = ctk.CTkLabel(
            self,
            text="Algorithm Comparison",
            font=("Segoe UI", 26, "bold"),
            text_color="#0F172A",
            anchor="w"
        )
        header.pack(padx=30, pady=(30, 5), anchor="w")
        
        subtitle = ctk.CTkLabel(
            self,
            text="Compare all clustering algorithms side-by-side using multiple performance metrics.",
            font=("Segoe UI", 14),
            text_color="#64748B",
            anchor="w"
        )
        subtitle.pack(padx=30, pady=(0, 20), anchor="w")

        # View Switcher
        self.view_var = ctk.StringVar(value="Configuration")
        self.view_switcher = ctk.CTkSegmentedButton(
            self, 
            values=["Configuration", "Results", "Visualizations"],
            variable=self.view_var,
            command=self.switch_view,
            font=("Segoe UI", 12, "bold"),
            height=32
        )
        self.view_switcher.pack(padx=30, pady=(0, 20), anchor="w")
        
        # Content Area
        self.content_area = ctk.CTkFrame(self, fg_color="transparent")
        self.content_area.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        
        # --- Configuration View ---
        self.config_frame = ctk.CTkFrame(
            self.content_area, 
            fg_color="white", 
            corner_radius=8, 
            border_width=1, 
            border_color="#E2E8F0"
        )
        self.create_config_view()
        
        # --- Results View ---
        self.results_frame = ctk.CTkFrame(
            self.content_area, 
            fg_color="white", 
            corner_radius=8, 
            border_width=1, 
            border_color="#E2E8F0"
        )
        self.create_results_view()
        
        # --- Visualizations View ---
        self.viz_frame = ctk.CTkFrame(
            self.content_area, 
            fg_color="white", 
            corner_radius=8, 
            border_width=1, 
            border_color="#E2E8F0"
        )
        self.create_viz_view()
        
        # Initialize view
        self.switch_view("Configuration")

    def switch_view(self, view_name):
        self.config_frame.pack_forget()
        self.results_frame.pack_forget()
        self.viz_frame.pack_forget()
        
        if view_name == "Configuration":
            self.config_frame.pack(fill="both", expand=True)
        elif view_name == "Results":
            self.results_frame.pack(fill="both", expand=True)
        elif view_name == "Visualizations":
            self.viz_frame.pack(fill="both", expand=True)

    def create_config_view(self):
        """Create the Configuration view"""
        config_scroll = ctk.CTkScrollableFrame(self.config_frame, fg_color="transparent")
        config_scroll.pack(fill="both", expand=True, padx=20, pady=20)
        
        inner = ctk.CTkFrame(config_scroll, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=20, pady=0)
        
        # Two column layout
        inner.grid_columnconfigure(0, weight=1)
        inner.grid_columnconfigure(1, weight=1)
        
        # Left Column - Algorithm Selection
        left_col = ctk.CTkFrame(inner, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        
        ctk.CTkLabel(
            left_col,
            text="Select Algorithms",
            font=("Segoe UI", 18, "bold"),
            text_color="#1E293B",
            anchor="w"
        ).pack(fill="x", pady=(0, 16))
        
        # Algorithm checkboxes - All 4 algorithms
        self.algo_vars = {
            "K-Means": ctk.BooleanVar(value=True),
            "K-Medoids": ctk.BooleanVar(value=True),
            "Hierarchical (AGNES)": ctk.BooleanVar(value=True),
            "DBSCAN": ctk.BooleanVar(value=True)
        }
        
        for algo, var in self.algo_vars.items():
            card = ctk.CTkFrame(left_col, fg_color="#F9FAFB", corner_radius=8)
            card.pack(fill="x", pady=4)
            
            ctk.CTkCheckBox(
                card, 
                text=algo, 
                variable=var,
                font=("Segoe UI", 13),
                fg_color="#3B82F6",
                hover_color="#2563EB",
                corner_radius=4
            ).pack(padx=16, pady=12, anchor="w")
        
        # Feature Selection - Multi-select with checkboxes
        ctk.CTkLabel(
            left_col,
            text="Feature Selection",
            font=("Segoe UI", 18, "bold"),
            text_color="#1E293B",
            anchor="w"
        ).pack(fill="x", pady=(24, 8))
        
        ctk.CTkLabel(
            left_col,
            text="Select multiple features. PCA used for visualization if >2 features.",
            font=("Segoe UI", 11),
            text_color="#64748B",
            anchor="w"
        ).pack(fill="x", pady=(0, 10))
        
        self.features_frame = ctk.CTkFrame(left_col, fg_color="#F9FAFB", corner_radius=8)
        self.features_frame.pack(fill="x", pady=(0, 10))
        
        self.feature_vars = {}
        self.feature_checkboxes_frame = ctk.CTkFrame(self.features_frame, fg_color="transparent")
        self.feature_checkboxes_frame.pack(fill="x", padx=15, pady=15)
        
        btn_row = ctk.CTkFrame(left_col, fg_color="transparent")
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
        
        # Auto-update features
        self.bind("<Visibility>", self.update_feature_options)
        
        # Right Column - Parameters
        right_col = ctk.CTkFrame(inner, fg_color="transparent")
        right_col.grid(row=0, column=1, sticky="nsew", padx=(20, 0))
        
        ctk.CTkLabel(
            right_col,
            text="Algorithm Parameters",
            font=("Segoe UI", 18, "bold"),
            text_color="#1E293B",
            anchor="w"
        ).pack(fill="x", pady=(0, 16))
        
        # K-Means/Hierarchical params
        params_card = ctk.CTkFrame(right_col, fg_color="#F9FAFB", corner_radius=8)
        params_card.pack(fill="x", pady=4)
        params_inner = ctk.CTkFrame(params_card, fg_color="transparent")
        params_inner.pack(fill="x", padx=16, pady=12)
        
        ctk.CTkLabel(params_inner, text="Number of Clusters (k)", font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x")
        ctk.CTkLabel(params_inner, text="For K-Means, K-Medoids, and Hierarchical", font=("Segoe UI", 11), text_color="#6B7280", anchor="w").pack(fill="x")
        self.k_entry = ctk.CTkEntry(params_inner, placeholder_text="e.g., 3", height=36)
        self.k_entry.insert(0, "3")
        self.k_entry.pack(fill="x", pady=(8, 0))
        
        # Hierarchical linkage
        linkage_card = ctk.CTkFrame(right_col, fg_color="#F9FAFB", corner_radius=8)
        linkage_card.pack(fill="x", pady=(16, 4))
        linkage_inner = ctk.CTkFrame(linkage_card, fg_color="transparent")
        linkage_inner.pack(fill="x", padx=16, pady=12)
        
        ctk.CTkLabel(linkage_inner, text="Hierarchical Linkage", font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x")
        self.linkage_var = ctk.StringVar(value="ward")
        ctk.CTkOptionMenu(
            linkage_inner, values=["ward", "complete", "average", "single"],
            variable=self.linkage_var, height=32
        ).pack(fill="x", pady=(8, 0))
        
        # DBSCAN params
        dbscan_card = ctk.CTkFrame(right_col, fg_color="#F9FAFB", corner_radius=8)
        dbscan_card.pack(fill="x", pady=(16, 4))
        dbscan_inner = ctk.CTkFrame(dbscan_card, fg_color="transparent")
        dbscan_inner.pack(fill="x", padx=16, pady=12)
        
        ctk.CTkLabel(dbscan_inner, text="DBSCAN Parameters", font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x")
        
        eps_frame = ctk.CTkFrame(dbscan_inner, fg_color="transparent")
        eps_frame.pack(fill="x", pady=(8, 4))
        ctk.CTkLabel(eps_frame, text="Epsilon (eps):", font=("Segoe UI", 11), width=120, anchor="w").pack(side="left")
        self.eps_entry = ctk.CTkEntry(eps_frame, width=100, height=32)
        self.eps_entry.insert(0, "0.5")
        self.eps_entry.pack(side="right")
        
        min_frame = ctk.CTkFrame(dbscan_inner, fg_color="transparent")
        min_frame.pack(fill="x", pady=4)
        ctk.CTkLabel(min_frame, text="Min Samples:", font=("Segoe UI", 11), width=120, anchor="w").pack(side="left")
        self.min_samples_entry = ctk.CTkEntry(min_frame, width=100, height=32)
        self.min_samples_entry.insert(0, "5")
        self.min_samples_entry.pack(side="right")
        
        # Progress and Run Button
        self.progress_bar = ctk.CTkProgressBar(right_col, mode="indeterminate", height=4)
        self.status_label = ctk.CTkLabel(right_col, text="", font=("Segoe UI", 12), text_color="#64748B")
        
        self.run_btn = ctk.CTkButton(
            right_col,
            text="🚀 Run Comparison",
            font=("Segoe UI", 14, "bold"),
            fg_color="#3B82F6",
            hover_color="#2563EB",
            height=48,
            command=self.run_comparison
        )
        self.run_btn.pack(fill="x", pady=(24, 0))

    def create_results_view(self):
        """Create the Results view with comparison table"""
        inner = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        ctk.CTkLabel(
            inner,
            text="Comparison Results",
            font=("Segoe UI", 18, "bold"),
            text_color="#1E293B",
            anchor="w"
        ).pack(fill="x", pady=(0, 8))
        
        ctk.CTkLabel(
            inner,
            text="Higher Silhouette & Calinski-Harabasz = Better | Lower Davies-Bouldin = Better",
            font=("Segoe UI", 12),
            text_color="#6B7280",
            anchor="w"
        ).pack(fill="x", pady=(0, 20))
        
        # Table container
        self.table_container = ctk.CTkScrollableFrame(inner, fg_color="transparent")
        self.table_container.pack(fill="both", expand=True)
        
        # Placeholder
        self.results_placeholder = ctk.CTkLabel(
            self.table_container,
            text="Run comparison to see results",
            font=("Segoe UI", 14),
            text_color="#94A3B8"
        )
        self.results_placeholder.pack(expand=True)
        
        # Winner section
        self.winner_card = ctk.CTkFrame(inner, fg_color="#ECFDF5", corner_radius=8)
        
        self.winner_label = ctk.CTkLabel(
            self.winner_card,
            text="",
            font=("Segoe UI", 14, "bold"),
            text_color="#059669"
        )
        self.winner_label.pack(padx=20, pady=16)

    def create_viz_view(self):
        """Create the Visualizations view"""
        # Controls at top
        viz_controls = ctk.CTkFrame(self.viz_frame, fg_color="transparent")
        viz_controls.pack(fill="x", padx=20, pady=(20, 10))
        
        ctk.CTkLabel(
            viz_controls,
            text="Visualization Type:",
            font=("Segoe UI", 12),
            anchor="w"
        ).pack(side="left", padx=(0, 10))
        
        self.viz_type_var = ctk.StringVar(value="Metrics Comparison")
        self.viz_type_menu = ctk.CTkOptionMenu(
            viz_controls,
            values=["Metrics Comparison", "Cluster Scatter Plots", "Cluster Distributions"],
            variable=self.viz_type_var,
            command=self.update_visualization,
            width=200
        )
        self.viz_type_menu.pack(side="left")
        
        self.pca_info_label = ctk.CTkLabel(
            viz_controls,
            text="",
            font=("Segoe UI", 11),
            text_color="#64748B"
        )
        self.pca_info_label.pack(side="right")
        
        # Plot container
        self.plot_container = ctk.CTkFrame(self.viz_frame, fg_color="transparent")
        self.plot_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Placeholder
        self.viz_placeholder = ctk.CTkLabel(
            self.plot_container,
            text="Run comparison to see visualizations",
            font=("Segoe UI", 14),
            text_color="#94A3B8"
        )
        self.viz_placeholder.pack(expand=True)

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
            var = ctk.BooleanVar(value=i < 2)  # Select first 2 by default
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

    def run_comparison(self):
        if self.is_running:
            return
            
        df = self.app.get_dataframe()
        if df is None:
            tk.messagebox.showwarning("No Data", "Please load a dataset first from the Data Loader page.")
            return
        
        selected_features = self.get_selected_features()
        if len(selected_features) < 2:
            tk.messagebox.showwarning("Insufficient Features", "Please select at least 2 features.")
            return
            
        # Validate inputs
        try:
            k = int(self.k_entry.get())
            eps = float(self.eps_entry.get())
            min_samples = int(self.min_samples_entry.get())
            linkage_method = self.linkage_var.get()
            
            if k < 2:
                tk.messagebox.showerror("Invalid Input", "Number of clusters must be at least 2.")
                return
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Please check your numeric inputs.")
            return
            
        selected_algos = [algo for algo, var in self.algo_vars.items() if var.get()]
        if not selected_algos:
            tk.messagebox.showwarning("No Selection", "Please select at least one algorithm.")
            return
        
        # Show progress
        self.is_running = True
        self.progress_bar.pack(fill="x", pady=(16, 8))
        self.progress_bar.start()
        self.status_label.configure(text="Running comparison...")
        self.status_label.pack(fill="x")
        self.run_btn.configure(state="disabled", text="Running...")
        
        # Run in thread
        thread = threading.Thread(
            target=self._run_comparison_thread, 
            args=(df, selected_algos, k, eps, min_samples, linkage_method, selected_features)
        )
        thread.daemon = True
        thread.start()

    def _run_comparison_thread(self, df, selected_algos, k, eps, min_samples, linkage_method, selected_features):
        try:
            X = df[selected_features].values
            X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
            
            # Sample if large
            if len(X) > 5000:
                sample_idx = np.random.choice(len(X), 5000, replace=False)
                X_sample = X[sample_idx]
            else:
                X_sample = X
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sample)
            
            # PCA for visualization if >2 features
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
            
            self.X_viz = X_viz
            self.viz_labels = viz_labels
                
            results = []
            
            for algo_name in selected_algos:
                labels = None
                centers = None
                
                if algo_name == "K-Means":
                    model = KMeans(n_clusters=k, n_init=10, random_state=42)
                    labels = model.fit_predict(X_scaled)
                    if len(selected_features) > 2:
                        centers = pca.transform(model.cluster_centers_)
                    else:
                        centers = scaler.inverse_transform(model.cluster_centers_)
                        
                elif algo_name == "K-Medoids":
                    model = KMeans(n_clusters=k, n_init=10, random_state=42)
                    model.fit(X_scaled)
                    closest, _ = pairwise_distances_argmin_min(model.cluster_centers_, X_scaled)
                    labels = model.predict(X_scaled)
                    if len(selected_features) > 2:
                        centers = X_viz[closest]
                    else:
                        centers = X_sample[closest]
                        
                elif algo_name == "Hierarchical (AGNES)":
                    model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
                    labels = model.fit_predict(X_scaled)
                    # Calculate centers per cluster
                    cluster_centers = []
                    for i in range(k):
                        mask = labels == i
                        if mask.sum() > 0:
                            cluster_centers.append(X_viz[mask].mean(axis=0))
                    centers = np.array(cluster_centers) if cluster_centers else None
                    
                elif algo_name == "DBSCAN":
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = model.fit_predict(X_scaled)
                    n_clusters_db = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters_db > 0:
                        cluster_centers = []
                        for i in range(n_clusters_db):
                            mask = labels == i
                            if mask.sum() > 0:
                                cluster_centers.append(X_viz[mask].mean(axis=0))
                        centers = np.array(cluster_centers) if cluster_centers else None
                    else:
                        centers = None
                
                # Calculate metrics
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                if n_clusters > 1:
                    if -1 in labels:
                        mask = labels != -1
                        X_metrics = X_scaled[mask]
                        labels_metrics = labels[mask]
                    else:
                        X_metrics = X_scaled
                        labels_metrics = labels
                        
                    if len(set(labels_metrics)) > 1:
                        sil = silhouette_score(X_metrics, labels_metrics)
                        db = davies_bouldin_score(X_metrics, labels_metrics)
                        ch = calinski_harabasz_score(X_metrics, labels_metrics)
                        
                        results.append({
                            "Algorithm": algo_name,
                            "Clusters": n_clusters,
                            "Silhouette": sil,
                            "Davies-Bouldin": db,
                            "Calinski-Harabasz": ch,
                            "Labels": labels,
                            "Centers": centers
                        })
                    else:
                        results.append({
                            "Algorithm": algo_name,
                            "Clusters": n_clusters,
                            "Silhouette": None,
                            "Davies-Bouldin": None,
                            "Calinski-Harabasz": None,
                            "Labels": labels,
                            "Centers": centers
                        })
                else:
                    results.append({
                        "Algorithm": algo_name,
                        "Clusters": n_clusters,
                        "Silhouette": None,
                        "Davies-Bouldin": None,
                        "Calinski-Harabasz": None,
                        "Labels": labels,
                        "Centers": centers
                    })
            
            self.after(0, lambda: self._finish_comparison(results, len(selected_features), explained_var))
            
        except Exception as e:
            self.after(0, lambda err=str(e): self._handle_error(err))

    def _finish_comparison(self, results, n_features, explained_var):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.configure(text="✓ Comparison complete!")
        self.after(2000, lambda: self.status_label.pack_forget())
        self.run_btn.configure(state="normal", text="🚀 Run Comparison")
        self.is_running = False
        
        self.results = results
        
        # Update PCA info
        if explained_var:
            self.pca_info_label.configure(text=f"📐 PCA: {explained_var:.1f}% variance from {n_features} features")
        else:
            self.pca_info_label.configure(text=f"📐 Using {n_features} features directly")
        
        self.display_results(results)
        self.update_visualization()
        
        # Auto-switch to Results view
        self.view_var.set("Results")
        self.switch_view("Results")

    def _handle_error(self, error_msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()
        self.run_btn.configure(state="normal", text="🚀 Run Comparison")
        self.is_running = False
        tk.messagebox.showerror("Error", f"Comparison failed: {error_msg}")

    def display_results(self, results):
        # Clear previous results
        for widget in self.table_container.winfo_children():
            widget.destroy()
            
        if not results:
            ctk.CTkLabel(self.table_container, text="No results to display.", text_color="#94A3B8").pack(expand=True)
            self.winner_card.pack_forget()
            return

        # Create header row
        header_frame = ctk.CTkFrame(self.table_container, fg_color="#F1F5F9", corner_radius=8)
        header_frame.pack(fill="x", pady=(0, 4))
        
        headers = ["Algorithm", "Clusters", "Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑"]
        header_inner = ctk.CTkFrame(header_frame, fg_color="transparent")
        header_inner.pack(fill="x", padx=16, pady=12)
        
        for i, header in enumerate(headers):
            ctk.CTkLabel(
                header_inner, 
                text=header, 
                font=("Segoe UI", 12, "bold"),
                text_color="#374151",
                width=140,
                anchor="w"
            ).pack(side="left", padx=4)
            
        # Find best performer
        valid_results = [r for r in results if r["Silhouette"] is not None]
        best_idx = -1
        if valid_results:
            best_sil = max(r["Silhouette"] for r in valid_results)
            for i, r in enumerate(results):
                if r["Silhouette"] == best_sil:
                    best_idx = i
                    break
        
        # Create data rows
        for i, row in enumerate(results):
            is_best = (i == best_idx)
            row_color = "#ECFDF5" if is_best else "#FFFFFF"
            border_color = "#10B981" if is_best else "#E5E7EB"
            
            row_frame = ctk.CTkFrame(self.table_container, fg_color=row_color, corner_radius=8, border_width=1, border_color=border_color)
            row_frame.pack(fill="x", pady=2)
            
            row_inner = ctk.CTkFrame(row_frame, fg_color="transparent")
            row_inner.pack(fill="x", padx=16, pady=12)
            
            # Algorithm name
            algo_frame = ctk.CTkFrame(row_inner, fg_color="transparent", width=140)
            algo_frame.pack(side="left", padx=4)
            algo_frame.pack_propagate(False)
            
            algo_text = "🏆 " + row["Algorithm"] if is_best else row["Algorithm"]
            ctk.CTkLabel(algo_frame, text=algo_text, font=("Segoe UI", 12, "bold" if is_best else "normal"), anchor="w").pack(anchor="w")
            
            # Clusters
            ctk.CTkLabel(row_inner, text=str(row["Clusters"]), font=("Segoe UI", 12), width=140, anchor="w").pack(side="left", padx=4)
            
            # Metrics
            sil_text = f"{row['Silhouette']:.4f}" if row["Silhouette"] is not None else "N/A"
            db_text = f"{row['Davies-Bouldin']:.4f}" if row["Davies-Bouldin"] is not None else "N/A"
            ch_text = f"{row['Calinski-Harabasz']:.1f}" if row["Calinski-Harabasz"] is not None else "N/A"
            
            ctk.CTkLabel(row_inner, text=sil_text, font=("Segoe UI", 12), width=140, anchor="w").pack(side="left", padx=4)
            ctk.CTkLabel(row_inner, text=db_text, font=("Segoe UI", 12), width=140, anchor="w").pack(side="left", padx=4)
            ctk.CTkLabel(row_inner, text=ch_text, font=("Segoe UI", 12), width=140, anchor="w").pack(side="left", padx=4)
        
        # Winner announcement
        if best_idx >= 0:
            winner = results[best_idx]
            self.winner_label.configure(
                text=f"🏆 Best Performer: {winner['Algorithm']} with Silhouette Score of {winner['Silhouette']:.4f}"
            )
            self.winner_card.pack(fill="x", pady=(20, 0))
        else:
            self.winner_card.pack_forget()

    def update_visualization(self, *args):
        """Update the visualization based on selected type"""
        if not self.results or self.X_viz is None:
            return
            
        for widget in self.plot_container.winfo_children():
            widget.destroy()
        
        viz_type = self.viz_type_var.get()
        
        if viz_type == "Metrics Comparison":
            self.plot_metrics_comparison()
        elif viz_type == "Cluster Scatter Plots":
            self.plot_cluster_scatter()
        elif viz_type == "Cluster Distributions":
            self.plot_cluster_distributions()

    def plot_metrics_comparison(self):
        valid_results = [r for r in self.results if r["Silhouette"] is not None]
        if not valid_results:
            ctk.CTkLabel(self.plot_container, text="No valid results to visualize", text_color="#94A3B8").pack(expand=True)
            return
        
        fig = Figure(figsize=(10, 4), dpi=100, facecolor='white')
        
        algos = [r["Algorithm"] for r in valid_results]
        x = np.arange(len(algos))
        
        ax1 = fig.add_subplot(131)
        sil_scores = [r["Silhouette"] for r in valid_results]
        bars1 = ax1.bar(x, sil_scores, color='#3B82F6', alpha=0.8)
        ax1.set_ylabel('Score')
        ax1.set_title('Silhouette Score', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([a[:10] for a in algos], rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        for bar, score in zip(bars1, sil_scores):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{score:.3f}',
                    ha='center', va='bottom', fontsize=8)
        
        ax2 = fig.add_subplot(132)
        db_scores = [r["Davies-Bouldin"] for r in valid_results]
        bars2 = ax2.bar(x, db_scores, color='#EF4444', alpha=0.8)
        ax2.set_ylabel('Score')
        ax2.set_title('Davies-Bouldin Index', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([a[:10] for a in algos], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        for bar, score in zip(bars2, db_scores):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{score:.3f}',
                    ha='center', va='bottom', fontsize=8)
        
        ax3 = fig.add_subplot(133)
        ch_scores = [r["Calinski-Harabasz"] for r in valid_results]
        bars3 = ax3.bar(x, ch_scores, color='#10B981', alpha=0.8)
        ax3.set_ylabel('Score')
        ax3.set_title('Calinski-Harabasz Index', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([a[:10] for a in algos], rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        for bar, score in zip(bars3, ch_scores):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{score:.0f}',
                    ha='center', va='bottom', fontsize=8)
        
        fig.tight_layout(pad=2)
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_cluster_scatter(self):
        valid_results = [r for r in self.results if r["Labels"] is not None]
        if not valid_results:
            ctk.CTkLabel(self.plot_container, text="No valid results to visualize", text_color="#94A3B8").pack(expand=True)
            return
        
        n_algos = len(valid_results)
        cols = min(n_algos, 2)
        rows = (n_algos + cols - 1) // cols
        
        fig = Figure(figsize=(5*cols, 4*rows), dpi=100, facecolor='white')
        
        for i, result in enumerate(valid_results):
            ax = fig.add_subplot(rows, cols, i + 1)
            
            labels = result["Labels"]
            centers = result["Centers"]
            
            unique_labels = set(labels)
            
            # Plot noise if exists
            if -1 in unique_labels:
                noise_mask = labels == -1
                ax.scatter(self.X_viz[noise_mask, 0], self.X_viz[noise_mask, 1], 
                          c='gray', marker='x', alpha=0.5, s=15, label='Noise')
            
            # Plot clusters
            cluster_mask = labels != -1
            if cluster_mask.any():
                scatter = ax.scatter(self.X_viz[cluster_mask, 0], self.X_viz[cluster_mask, 1], 
                                   c=labels[cluster_mask], cmap='viridis', alpha=0.6, s=20)
            
            # Plot centers
            if centers is not None and len(centers) > 0:
                ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', 
                          s=100, edgecolors='black', linewidth=1, zorder=5, label='Centers')
            
            sil_text = f"Sil: {result['Silhouette']:.3f}" if result['Silhouette'] else "Sil: N/A"
            ax.set_title(f"{result['Algorithm']}\n{sil_text}", fontweight='bold', fontsize=10)
            ax.set_xlabel(self.viz_labels[0], fontsize=9)
            ax.set_ylabel(self.viz_labels[1], fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.set_facecolor('#FAFAFA')
        
        fig.tight_layout(pad=2)
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_cluster_distributions(self):
        valid_results = [r for r in self.results if r["Labels"] is not None]
        if not valid_results:
            ctk.CTkLabel(self.plot_container, text="No valid results to visualize", text_color="#94A3B8").pack(expand=True)
            return
        
        n_algos = len(valid_results)
        cols = min(n_algos, 2)
        rows = (n_algos + cols - 1) // cols
        
        fig = Figure(figsize=(5*cols, 4*rows), dpi=100, facecolor='white')
        
        colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899']
        
        for i, result in enumerate(valid_results):
            ax = fig.add_subplot(rows, cols, i + 1)
            
            labels = result["Labels"]
            unique, counts = np.unique(labels, return_counts=True)
            
            bar_labels = []
            bar_counts = []
            bar_colors = []
            
            color_idx = 0
            for u, c in zip(unique, counts):
                if u == -1:
                    bar_labels.append('Noise')
                    bar_colors.append('gray')
                else:
                    bar_labels.append(f'C{u}')
                    bar_colors.append(colors[color_idx % len(colors)])
                    color_idx += 1
                bar_counts.append(c)
            
            bars = ax.bar(bar_labels, bar_counts, color=bar_colors, alpha=0.8)
            
            for bar, count in zip(bars, bar_counts):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{count}',
                       ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Cluster', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f"{result['Algorithm']}\n({result['Clusters']} clusters)", fontweight='bold', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.set_facecolor('#FAFAFA')
        
        fig.tight_layout(pad=2)
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
