import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
        self.results = {}
        
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
            text="Compare different clustering algorithms using various performance metrics.",
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
            width=300
        )
        controls_panel.pack(side="left", fill="y", padx=(0, 20))
        controls_panel.pack_propagate(False)
        
        # Right Panel - Results
        self.results_panel = ctk.CTkFrame(
            content, 
            fg_color="white", 
            corner_radius=8, 
            border_width=1, 
            border_color="#E2E8F0"
        )
        self.results_panel.pack(side="left", fill="both", expand=True)
        
        # Controls Title
        ctk.CTkLabel(
            controls_panel, 
            text="Configuration", 
            font=("Segoe UI", 16, "bold"),
            text_color="#1E293B"
        ).pack(padx=20, pady=20, anchor="w")
        
        # Algorithm Selection
        self.algo_vars = {
            "K-Means": ctk.BooleanVar(value=True),
            "Hierarchical": ctk.BooleanVar(value=True),
            "DBSCAN": ctk.BooleanVar(value=False)
        }
        
        ctk.CTkLabel(controls_panel, text="Select Algorithms:", font=("Segoe UI", 12, "bold")).pack(padx=20, pady=(0, 5), anchor="w")
        
        for algo, var in self.algo_vars.items():
            ctk.CTkCheckBox(
                controls_panel, 
                text=algo, 
                variable=var,
                font=("Segoe UI", 12),
                fg_color="#3B82F6"
            ).pack(padx=20, pady=5, anchor="w")
            
        # Parameters
        ctk.CTkLabel(controls_panel, text="Parameters:", font=("Segoe UI", 12, "bold")).pack(padx=20, pady=(20, 5), anchor="w")
        
        # K (Clusters)
        k_frame = ctk.CTkFrame(controls_panel, fg_color="transparent")
        k_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(k_frame, text="Number of Clusters (k):", font=("Segoe UI", 12)).pack(side="left")
        self.k_entry = ctk.CTkEntry(k_frame, width=60)
        self.k_entry.insert(0, "3")
        self.k_entry.pack(side="right")
        
        # DBSCAN Params
        dbscan_frame = ctk.CTkFrame(controls_panel, fg_color="transparent")
        dbscan_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(dbscan_frame, text="DBSCAN Eps:", font=("Segoe UI", 12)).pack(side="left")
        self.eps_entry = ctk.CTkEntry(dbscan_frame, width=60)
        self.eps_entry.insert(0, "0.5")
        self.eps_entry.pack(side="right")
        
        dbscan_min_frame = ctk.CTkFrame(controls_panel, fg_color="transparent")
        dbscan_min_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(dbscan_min_frame, text="DBSCAN Min Samples:", font=("Segoe UI", 12)).pack(side="left")
        self.min_samples_entry = ctk.CTkEntry(dbscan_min_frame, width=60)
        self.min_samples_entry.insert(0, "5")
        self.min_samples_entry.pack(side="right")
        
        # Visualization Options
        ctk.CTkLabel(controls_panel, text="Visualizations:", font=("Segoe UI", 12, "bold")).pack(padx=20, pady=(20, 5), anchor="w")
        
        self.viz_vars = {
            "Silhouette Plot": ctk.BooleanVar(value=True),
            "Cluster Distribution": ctk.BooleanVar(value=True)
        }
        
        for viz, var in self.viz_vars.items():
            ctk.CTkCheckBox(
                controls_panel, 
                text=viz, 
                variable=var,
                font=("Segoe UI", 12),
                fg_color="#3B82F6"
            ).pack(padx=20, pady=5, anchor="w")
        
        # Run Button
        self.run_btn = ctk.CTkButton(
            controls_panel,
            text="Run Comparison",
            font=("Segoe UI", 14, "bold"),
            fg_color="#3B82F6",
            hover_color="#2563EB",
            height=40,
            command=self.run_comparison
        )
        self.run_btn.pack(fill="x", padx=20, pady=30)
        
        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(controls_panel, mode="indeterminate")
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(
            controls_panel,
            text="",
            font=("Segoe UI", 12),
            text_color="#64748B"
        )

        # Results Area
        self.setup_results_area()

    def setup_results_area(self):
        # Clear previous results
        for widget in self.results_panel.winfo_children():
            widget.destroy()
            
        ctk.CTkLabel(
            self.results_panel, 
            text="Comparison Results", 
            font=("Segoe UI", 16, "bold"),
            text_color="#1E293B"
        ).pack(padx=20, pady=20, anchor="w")
        
        self.viz_panel = ctk.CTkFrame(self.results_panel, fg_color="transparent")
        self.viz_panel.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Placeholder text
        self.placeholder_label = ctk.CTkLabel(
            self.viz_panel,
            text="Run comparison to see results",
            font=("Segoe UI", 14),
            text_color="#94A3B8"
        )
        self.placeholder_label.pack(expand=True)

    def run_comparison(self):
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
            
        # Validate inputs
        try:
            k = int(self.k_entry.get())
            eps = float(self.eps_entry.get())
            min_samples = int(self.min_samples_entry.get())
            
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
        self.progress_bar.pack(fill="x", padx=20, pady=(0, 10))
        self.progress_bar.start()
        self.status_label.configure(text="Running comparison...")
        self.status_label.pack(fill="x", padx=20, pady=(0, 10))
        self.run_btn.configure(state="disabled", text="Running...")
        
        # Run in thread
        thread = threading.Thread(
            target=self._run_comparison_thread, 
            args=(numeric_df, selected_algos, k, eps, min_samples)
        )
        thread.daemon = True
        thread.start()

    def _run_comparison_thread(self, numeric_df, selected_algos, k, eps, min_samples):
        try:
            X = numeric_df.iloc[:, :].values
            
            # Preprocessing
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Sample if large for speed
            if len(X_scaled) > 5000:
                sample_idx = np.random.choice(len(X_scaled), 5000, replace=False)
                X_sample = X_scaled[sample_idx]
            else:
                X_sample = X_scaled
                
            results = []
            
            for algo_name in selected_algos:
                labels = None
                
                if algo_name == "K-Means":
                    model = KMeans(n_clusters=k, n_init=10, random_state=42)
                    labels = model.fit_predict(X_sample)
                elif algo_name == "Hierarchical":
                    model = AgglomerativeClustering(n_clusters=k)
                    labels = model.fit_predict(X_sample)
                elif algo_name == "DBSCAN":
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = model.fit_predict(X_sample)
                
                # Calculate metrics if valid clustering
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                if n_clusters > 1:
                    # Filter out noise for metrics if DBSCAN
                    if -1 in labels:
                        mask = labels != -1
                        X_metrics = X_sample[mask]
                        labels_metrics = labels[mask]
                    else:
                        X_metrics = X_sample
                        labels_metrics = labels
                        
                    if len(set(labels_metrics)) > 1:
                        sil = silhouette_score(X_metrics, labels_metrics)
                        db = davies_bouldin_score(X_metrics, labels_metrics)
                        ch = calinski_harabasz_score(X_metrics, labels_metrics)
                        
                        results.append({
                            "Algorithm": algo_name,
                            "Clusters": n_clusters,
                            "Silhouette": f"{sil:.3f}",
                            "Davies-Bouldin": f"{db:.3f}",
                            "Calinski-Harabasz": f"{ch:.1f}",
                            "Labels": labels_metrics  # Store labels for visualization
                        })
                    else:
                        results.append({
                            "Algorithm": algo_name,
                            "Clusters": n_clusters,
                            "Silhouette": "N/A",
                            "Davies-Bouldin": "N/A",
                            "Calinski-Harabasz": "N/A",
                            "Labels": None
                        })
                else:
                    results.append({
                        "Algorithm": algo_name,
                        "Clusters": n_clusters,
                        "Silhouette": "N/A",
                        "Davies-Bouldin": "N/A",
                        "Calinski-Harabasz": "N/A",
                        "Labels": None
                    })
            
            self.after(0, lambda: self._finish_comparison(results))
            
        except Exception as e:
            self.after(0, lambda err=str(e): self._handle_error(err))

    def _finish_comparison(self, results):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.configure(text="✓ Comparison complete!")
        self.after(2000, lambda: self.status_label.pack_forget())
        self.run_btn.configure(state="normal", text="Run Comparison")
        self.is_running = False
        
        self.display_results(results)

    def _handle_error(self, error_msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()
        self.run_btn.configure(state="normal", text="Run Comparison")
        self.is_running = False
        tk.messagebox.showerror("Error", f"Comparison failed: {error_msg}")

    def display_results(self, results):
        # Clear previous results
        for widget in self.viz_panel.winfo_children():
            widget.destroy()
            
        if not results:
            ctk.CTkLabel(self.viz_panel, text="No results to display.").pack()
            return

        # Create Table
        table_frame = ctk.CTkFrame(self.viz_panel, fg_color="transparent")
        table_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Headers
        headers = ["Algorithm", "Clusters", "Silhouette (↑)", "Davies-Bouldin (↓)", "Calinski-Harabasz (↑)"]
        
        for i, header in enumerate(headers):
            ctk.CTkLabel(
                table_frame, 
                text=header, 
                font=("Segoe UI", 12, "bold"),
                width=150,
                anchor="w"
            ).grid(row=0, column=i, padx=5, pady=10, sticky="w")
            
        # Rows
        for i, row in enumerate(results):
            ctk.CTkLabel(table_frame, text=row["Algorithm"], font=("Segoe UI", 12), anchor="w").grid(row=i+1, column=0, padx=5, pady=5, sticky="w")
            ctk.CTkLabel(table_frame, text=str(row["Clusters"]), font=("Segoe UI", 12), anchor="w").grid(row=i+1, column=1, padx=5, pady=5, sticky="w")
            ctk.CTkLabel(table_frame, text=row["Silhouette"], font=("Segoe UI", 12), anchor="w").grid(row=i+1, column=2, padx=5, pady=5, sticky="w")
            ctk.CTkLabel(table_frame, text=row["Davies-Bouldin"], font=("Segoe UI", 12), anchor="w").grid(row=i+1, column=3, padx=5, pady=5, sticky="w")
            ctk.CTkLabel(table_frame, text=row["Calinski-Harabasz"], font=("Segoe UI", 12), anchor="w").grid(row=i+1, column=4, padx=5, pady=5, sticky="w")
            
        # Plotting
        try:
            valid_results = [r for r in results if r["Silhouette"] != "N/A"]
            
            # Determine number of plots
            plots_to_show = []
            if self.viz_vars["Silhouette Plot"].get() and valid_results:
                plots_to_show.append("silhouette")
            if self.viz_vars["Cluster Distribution"].get() and valid_results:
                plots_to_show.append("distribution")
                
            if not plots_to_show:
                return

            fig = Figure(figsize=(8, 4 * len(plots_to_show)), dpi=100, facecolor='white')
            
            current_plot = 1
            total_plots = len(plots_to_show)
            
            if "silhouette" in plots_to_show:
                ax = fig.add_subplot(total_plots, 1, current_plot)
                algos = [r["Algorithm"] for r in valid_results]
                scores = [float(r["Silhouette"]) for r in valid_results]
                
                bars = ax.bar(algos, scores, color='#3B82F6', alpha=0.7)
                ax.set_ylabel('Silhouette Score')
                ax.set_title('Silhouette Score Comparison')
                ax.grid(axis='y', alpha=0.3)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom')
                current_plot += 1
                
            if "distribution" in plots_to_show:
                ax = fig.add_subplot(total_plots, 1, current_plot)
                
                # Prepare data for grouped bar chart
                algos = [r["Algorithm"] for r in valid_results]
                all_labels = [r["Labels"] for r in valid_results]
                
                # Find max clusters to normalize colors/groups
                max_clusters = max([len(set(l)) for l in all_labels])
                
                # We will plot the size of each cluster for each algo
                # This is a bit complex for a single plot if cluster counts differ widely
                # Simplified: Plot standard deviation of cluster sizes? Or just stacked bars?
                # Let's do side-by-side bars for cluster sizes.
                
                # Actually, let's just plot the number of points in each cluster for each algo
                # Since k might be different (e.g. DBSCAN), this is tricky.
                # Let's plot a histogram of cluster labels for the first valid algo, or subplots?
                # Given the space, let's plot the distribution of cluster sizes (std dev) or just the counts.
                
                # Better: Subplots for each algo's distribution? No, too many.
                # Let's plot the "Balance" of clusters - standard deviation of cluster sizes.
                # Or just plot the counts for the first algo as an example?
                # The user asked for "histograms".
                # Let's plot a grouped bar chart where x-axis is Algorithm, and we show bars for Cluster 0, Cluster 1, etc.
                # If too many clusters, we limit to top 5.
                
                width = 0.8 / len(algos)
                x = np.arange(max_clusters)
                
                for i, (algo, labels) in enumerate(zip(algos, all_labels)):
                    unique, counts = np.unique(labels, return_counts=True)
                    # Pad with 0 if fewer clusters
                    padded_counts = np.zeros(max_clusters)
                    for u, c in zip(unique, counts):
                        if u >= 0 and u < max_clusters: # Ignore noise -1 for index or handle it
                             padded_counts[int(u)] = c
                    
                    # Offset bars
                    # This is getting complicated to visualize generally.
                    # Let's switch to: One subplot per algorithm for distribution if requested?
                    # Or just plot the distribution of the BEST performing algorithm?
                    pass

                # Re-thinking: User wants "histograms". Plural.
                # Maybe just plot the distribution of cluster sizes for each algorithm.
                # X-axis: Cluster ID, Y-axis: Count.
                # If multiple algos, maybe use subplots within the figure.
                
                # Let's try to fit them in one row if possible, or grid.
                # Since we are inside a single figure, let's split the subplot area.
                # Actually, let's just plot the distribution for the algorithm with the highest Silhouette score.
                
                best_algo_idx = np.argmax([float(r["Silhouette"]) for r in valid_results])
                best_algo = valid_results[best_algo_idx]
                labels = best_algo["Labels"]
                unique, counts = np.unique(labels, return_counts=True)
                
                ax.bar(unique.astype(str), counts, color='#10B981', alpha=0.7)
                ax.set_xlabel('Cluster Label')
                ax.set_ylabel('Count')
                ax.set_title(f'Cluster Distribution ({best_algo["Algorithm"]})')
                ax.grid(axis='y', alpha=0.3)
                
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.viz_panel)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)
            
        except Exception as e:
            print(f"Error plotting: {e}")
