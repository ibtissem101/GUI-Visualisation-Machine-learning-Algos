import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import seaborn as sns
import threading

class EDAPage(ctk.CTkFrame):
    def __init__(self, master, app_instance):
        super().__init__(master, fg_color="#F5F5F5", corner_radius=0)
        self.app = app_instance
        self.canvas = None
        self.is_generating = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header = ctk.CTkLabel(
            self,
            text="Exploratory Data Analysis",
            font=("Segoe UI", 26, "bold"),
            text_color="#0F172A",
            anchor="w"
        )
        header.pack(padx=30, pady=(30, 5), anchor="w")
        
        subtitle = ctk.CTkLabel(
            self,
            text="Visualize and explore your dataset with statistical summaries and charts.",
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
        
        # Controls content
        controls_inner = ctk.CTkFrame(controls_panel, fg_color="transparent")
        controls_inner.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(
            controls_inner, 
            text="Visualizations", 
            font=("Segoe UI", 16, "bold"), 
            text_color="#1E293B",
            anchor="w"
        ).pack(fill="x", pady=(0, 20))
        
        # Chart type selection
        ctk.CTkLabel(
            controls_inner, 
            text="Chart Type", 
            text_color="#475569",
            font=("Segoe UI", 12),
            anchor="w"
        ).pack(fill="x", pady=(0, 8))
        
        self.chart_var = ctk.StringVar(value="Statistics Summary")
        chart_types = [
            "Statistics Summary",
            "Distribution Plots", 
            "Correlation Heatmap",
            "Box Plots",
            "Scatter Matrix",
            "Missing Data",
            "Pairwise Correlations"
        ]
        
        self.chart_menu = ctk.CTkOptionMenu(
            controls_inner,
            values=chart_types,
            variable=self.chart_var,
            fg_color="#2D5BFF",
            button_color="#2D5BFF",
            button_hover_color="#1E40AF",
            dropdown_fg_color="white",
            height=36,
            font=("Segoe UI", 12)
        )
        self.chart_menu.pack(fill="x", pady=(0, 20))
        
        # Info box
        info_box = ctk.CTkFrame(controls_inner, fg_color="#EFF6FF", corner_radius=6)
        info_box.pack(fill="x", pady=(10, 20))
        
        ctk.CTkLabel(
            info_box, 
            text="ℹ️  Select a visualization type\nto explore your data.", 
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
        
        # Generate Button
        self.generate_btn = ctk.CTkButton(
            controls_inner,
            text="Generate Visualization",
            command=self.generate_viz,
            font=("Segoe UI", 13, "bold"),
            fg_color="#2D5BFF",
            hover_color="#1E40AF",
            height=44
        )
        self.generate_btn.pack(fill="x", pady=(0, 0))
        
        # Right Panel - Visualization
        self.viz_panel = ctk.CTkScrollableFrame(
            content, 
            fg_color="white", 
            corner_radius=8, 
            border_width=1, 
            border_color="#E2E8F0"
        )
        self.viz_panel.pack(side="right", fill="both", expand=True)
        
        # Initial message
        self.plot_placeholder()

    def plot_placeholder(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
            
        placeholder = ctk.CTkLabel(
            self.viz_panel,
            text="Load data and select a visualization type\nto see EDA charts",
            text_color="#64748B",
            font=("Segoe UI", 14)
        )
        placeholder.pack(expand=True, pady=100)

    def generate_viz(self):
        if self.is_generating:
            return
            
        df = self.app.get_dataframe()
        if df is None:
            tk.messagebox.showwarning("No Data", "Please load a dataset first from the Data Loader page.")
            return
        
        chart_type = self.chart_var.get()
        
        # Show progress
        self.is_generating = True
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.start()
        self.status_label.configure(text=f"Generating {chart_type}...")
        self.status_label.pack(fill="x", pady=(0, 10))
        self.generate_btn.configure(state="disabled", text="Generating...")
        
        # Run in thread
        thread = threading.Thread(target=self._generate_viz_thread, args=(df.copy(), chart_type))
        thread.daemon = True
        thread.start()
    
    def _generate_viz_thread(self, df, chart_type):
        try:
            # Generate based on type
            result = None
            if chart_type == "Statistics Summary":
                result = ('stats', df)
            elif chart_type == "Distribution Plots":
                result = ('plot', self._create_distributions(df))
            elif chart_type == "Correlation Heatmap":
                result = ('plot', self._create_correlation(df))
            elif chart_type == "Box Plots":
                result = ('plot', self._create_boxplots(df))
            elif chart_type == "Scatter Matrix":
                result = ('plot', self._create_scatter_matrix(df))
            elif chart_type == "Missing Data":
                result = ('plot', self._create_missing_data(df))
            elif chart_type == "Pairwise Correlations":
                result = ('plot', self._create_pairwise_corr(df))
            
            self.after(0, lambda: self._finish_viz(result, chart_type))
        except Exception as e:
            self.after(0, lambda err=str(e): self._handle_viz_error(err))
    
    def _finish_viz(self, result, chart_type):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.configure(text="✓ Visualization complete!")
        self.after(2000, lambda: self.status_label.pack_forget())
        self.generate_btn.configure(state="normal", text="Generate Visualization")
        self.is_generating = False
        
        # Clear previous
        for widget in self.viz_panel.winfo_children():
            widget.destroy()
        
        if result:
            viz_type, data = result
            if viz_type == 'stats':
                self.show_statistics(data)
            elif viz_type == 'plot':
                self._display_plot(data)
    
    def _handle_viz_error(self, error_msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()
        self.generate_btn.configure(state="normal", text="Generate Visualization")
        self.is_generating = False
        tk.messagebox.showerror("Visualization Error", f"Error generating visualization: {error_msg}")
    
    def _display_plot(self, fig):
        if fig:
            self.canvas = FigureCanvasTkAgg(fig, master=self.viz_panel)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)

    def show_statistics(self, df):
        """Show statistical summary"""
        stats_frame = ctk.CTkFrame(self.viz_panel, fg_color="transparent")
        stats_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Numeric summary
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            ctk.CTkLabel(
                stats_frame,
                text="Numeric Features Summary",
                font=("Segoe UI", 16, "bold"),
                text_color="#1E293B",
                anchor="w"
            ).pack(fill="x", pady=(0, 15))
            
            stats = numeric_df.describe()
            self.create_stats_table(stats_frame, stats)
        
        # Categorical summary
        cat_df = df.select_dtypes(include=['object'])
        if len(cat_df.columns) > 0:
            ctk.CTkLabel(
                stats_frame,
                text="Categorical Features Summary",
                font=("Segoe UI", 16, "bold"),
                text_color="#1E293B",
                anchor="w"
            ).pack(fill="x", pady=(20, 15))
            
            for col in cat_df.columns[:5]:  # Show first 5 categorical
                value_counts = df[col].value_counts().head(5)
                self.create_value_count_card(stats_frame, col, value_counts)

    def create_stats_table(self, parent, stats):
        """Create a styled statistics table"""
        table_frame = ctk.CTkFrame(parent, fg_color="#F8FAFC", corner_radius=8)
        table_frame.pack(fill="x", pady=(0, 10))
        
        # Headers
        headers = ["Statistic"] + list(stats.columns)
        for i, header in enumerate(headers):
            label = ctk.CTkLabel(
                table_frame,
                text=header if i == 0 else header[:15],
                font=("Segoe UI", 11, "bold"),
                text_color="#1E293B",
                anchor="w"
            )
            label.grid(row=0, column=i, padx=10, pady=10, sticky="w")
        
        # Data
        for i, idx in enumerate(stats.index):
            # Row label
            ctk.CTkLabel(
                table_frame,
                text=idx,
                font=("Segoe UI", 10),
                text_color="#64748B",
                anchor="w"
            ).grid(row=i+1, column=0, padx=10, pady=5, sticky="w")
            
            # Values
            for j, col in enumerate(stats.columns):
                val = stats.loc[idx, col]
                ctk.CTkLabel(
                    table_frame,
                    text=f"{val:.2f}" if not pd.isna(val) else "N/A",
                    font=("Segoe UI", 10),
                    text_color="#475569",
                    anchor="w"
                ).grid(row=i+1, column=j+1, padx=10, pady=5, sticky="w")

    def create_value_count_card(self, parent, column_name, value_counts):
        """Create a card showing value counts"""
        card = ctk.CTkFrame(parent, fg_color="#F8FAFC", corner_radius=8, border_width=1, border_color="#E2E8F0")
        card.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            card,
            text=column_name,
            font=("Segoe UI", 12, "bold"),
            text_color="#1E293B",
            anchor="w"
        ).pack(padx=15, pady=(10, 5), anchor="w")
        
        for val, count in value_counts.items():
            ctk.CTkLabel(
                card,
                text=f"  • {val}: {count}",
                font=("Segoe UI", 10),
                text_color="#64748B",
                anchor="w"
            ).pack(padx=15, pady=2, anchor="w")
        
        ctk.CTkFrame(card, fg_color="transparent", height=5).pack()

    def _create_distributions(self, df):
        """Create distribution plots figure"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) == 0:
            raise ValueError("No numeric columns to visualize")
        
        n_cols = min(len(numeric_df.columns), 6)
        n_rows = (n_cols + 1) // 2
        
        fig = Figure(figsize=(12, 4*n_rows), dpi=100, facecolor='white')
        
        for i, col in enumerate(numeric_df.columns[:6]):
            ax = fig.add_subplot(n_rows, 2, i+1)
            ax.hist(numeric_df[col].dropna(), bins=30, alpha=0.7, color='#2D5BFF', edgecolor='white')
            ax.set_title(f'{col} Distribution', fontweight='bold', fontsize=11)
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_facecolor('#FAFAFA')
        
        fig.tight_layout()
        return fig

    def show_distributions(self, df):
        """Show distribution plots for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) == 0:
            tk.messagebox.showinfo("No Numeric Data", "No numeric columns to visualize.")
            return
        
        n_cols = min(len(numeric_df.columns), 6)  # Max 6 plots
        n_rows = (n_cols + 1) // 2
        
        fig = Figure(figsize=(12, 4*n_rows), dpi=100, facecolor='white')
        
        for i, col in enumerate(numeric_df.columns[:6]):
            ax = fig.add_subplot(n_rows, 2, i+1)
            ax.hist(numeric_df[col].dropna(), bins=30, alpha=0.7, color='#2D5BFF', edgecolor='white')
            ax.set_title(f'{col} Distribution', fontweight='bold', fontsize=11)
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_facecolor('#FAFAFA')
        
        fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)

    def show_correlation(self, df):
        """Show correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            tk.messagebox.showinfo("Insufficient Data", "Need at least 2 numeric columns for correlation.")
            return
        
        corr = numeric_df.corr()
        
        fig = Figure(figsize=(10, 8), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        im = ax.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(corr.columns, fontsize=9)
        
        # Add correlation values
        for i in range(len(corr)):
            for j in range(len(corr)):
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Correlation Heatmap', fontweight='bold', fontsize=13, pad=15)
        fig.colorbar(im, ax=ax, label='Correlation')
        fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)

    def show_boxplots(self, df):
        """Show box plots for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) == 0:
            tk.messagebox.showinfo("No Numeric Data", "No numeric columns to visualize.")
            return
        
        n_cols = min(len(numeric_df.columns), 6)
        n_rows = (n_cols + 1) // 2
        
        fig = Figure(figsize=(12, 4*n_rows), dpi=100, facecolor='white')
        
        for i, col in enumerate(numeric_df.columns[:6]):
            ax = fig.add_subplot(n_rows, 2, i+1)
            bp = ax.boxplot(numeric_df[col].dropna(), vert=True, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#2D5BFF')
                patch.set_alpha(0.7)
            ax.set_title(f'{col} Box Plot', fontweight='bold', fontsize=11)
            ax.set_ylabel(col, fontsize=10)
            ax.grid(True, alpha=0.2, linestyle='--', axis='y')
            ax.set_facecolor('#FAFAFA')
        
        fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)

    def show_scatter_matrix(self, df):
        """Show scatter matrix for first few numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            tk.messagebox.showinfo("Insufficient Data", "Need at least 2 numeric columns.")
            return
        
        # Use first 4 columns max
        cols_to_use = numeric_df.columns[:4]
        n = len(cols_to_use)
        
        fig = Figure(figsize=(12, 12), dpi=100, facecolor='white')
        
        for i in range(n):
            for j in range(n):
                ax = fig.add_subplot(n, n, i*n + j + 1)
                if i == j:
                    # Diagonal: histogram
                    ax.hist(numeric_df[cols_to_use[i]].dropna(), bins=20, alpha=0.7, color='#2D5BFF', edgecolor='white')
                else:
                    # Off-diagonal: scatter
                    ax.scatter(numeric_df[cols_to_use[j]], numeric_df[cols_to_use[i]], 
                             alpha=0.5, s=10, color='#2D5BFF')
                
                if i == n-1:
                    ax.set_xlabel(cols_to_use[j], fontsize=9)
                else:
                    ax.set_xticklabels([])
                
                if j == 0:
                    ax.set_ylabel(cols_to_use[i], fontsize=9)
                else:
                    ax.set_yticklabels([])
                
                ax.grid(True, alpha=0.2)
                ax.set_facecolor('#FAFAFA')
        
        fig.suptitle('Scatter Matrix', fontweight='bold', fontsize=14, y=0.995)
        fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)

    def show_missing_data(self, df):
        """Visualize missing data"""
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            tk.messagebox.showinfo("No Missing Data", "No missing values found in the dataset!")
            return
        
        fig = Figure(figsize=(10, max(6, len(missing)*0.4)), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        y_pos = range(len(missing))
        ax.barh(y_pos, missing.values, color='#EF4444', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(missing.index, fontsize=10)
        ax.set_xlabel('Number of Missing Values', fontsize=11, fontweight='bold')
        ax.set_title('Missing Data by Column', fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.2, linestyle='--', axis='x')
        ax.set_facecolor('#FAFAFA')
        
        fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)

    def show_pairwise_corr(self, df):
        """Show pairwise correlations as bar chart"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            tk.messagebox.showinfo("Insufficient Data", "Need at least 2 numeric columns.")
            return
        
        corr = numeric_df.corr()
        # Get upper triangle
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_pairs.append((f"{corr.columns[i]} vs {corr.columns[j]}", corr.iloc[i, j]))
        
        corr_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        corr_pairs = corr_pairs[:15]  # Top 15
        
        labels, values = zip(*corr_pairs)
        
        fig = Figure(figsize=(10, max(6, len(labels)*0.4)), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        colors = ['#10B981' if v > 0 else '#EF4444' for v in values]
        y_pos = range(len(labels))
        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Correlation Coefficient', fontsize=11, fontweight='bold')
        ax.set_title('Top Pairwise Correlations', fontsize=13, fontweight='bold', pad=15)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.2, linestyle='--', axis='x')
        ax.set_facecolor('#FAFAFA')
        ax.set_xlim(-1, 1)
        
        fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.viz_panel)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)
