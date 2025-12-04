import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import seaborn as sns
import threading

class EDAPage(ctk.CTkFrame):
    def __init__(self, master, app_instance):
        super().__init__(master, fg_color="#F9FAFB", corner_radius=0)
        self.app = app_instance
        self.canvas = None
        self.is_generating = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Header with consistent padding
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(padx=40, pady=(40, 0), fill="x")
        
        header = ctk.CTkLabel(
            header_frame,
            text="Exploratory Data Analysis",
            font=("Segoe UI", 28, "bold"),
            text_color="#111827",
            anchor="w"
        )
        header.pack(anchor="w")
        
        subtitle = ctk.CTkLabel(
            header_frame,
            text="Visualize and explore your dataset with statistical summaries and charts",
            font=("Segoe UI", 14),
            text_color="#6B7280",
            anchor="w"
        )
        subtitle.pack(anchor="w", pady=(6, 0))
        
        # Divider
        divider = ctk.CTkFrame(self, fg_color="#E5E7EB", height=1)
        divider.pack(fill="x", padx=40, pady=(25, 30))
        
        # Content Layout
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=40, pady=(0, 40))
        
        # Left Panel - Controls
        controls_panel = ctk.CTkFrame(
            content, 
            fg_color="white", 
            corner_radius=12, 
            border_width=1, 
            border_color="#E5E7EB"
        )
        controls_panel.pack(side="left", fill="y", padx=(0, 15), ipadx=130)
        controls_panel.pack_propagate(False)
        
        # Controls content
        controls_inner = ctk.CTkFrame(controls_panel, fg_color="transparent")
        controls_inner.pack(fill="both", expand=True, padx=24, pady=24)
        
        ctk.CTkLabel(
            controls_inner, 
            text="Visualizations", 
            font=("Segoe UI", 16, "bold"), 
            text_color="#111827",
            anchor="w"
        ).pack(fill="x", pady=(0, 20))
        
        # Chart type selection
        ctk.CTkLabel(
            controls_inner, 
            text="Chart Type", 
            text_color="#6B7280",
            font=("Segoe UI", 12),
            anchor="w"
        ).pack(fill="x", pady=(0, 10))
        
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
            fg_color="#2563EB",
            button_color="#2563EB",
            button_hover_color="#1D4ED8",
            dropdown_fg_color="white",
            dropdown_hover_color="#F3F4F6",
            dropdown_text_color="#111827",
            height=40,
            font=("Segoe UI", 12),
            corner_radius=8
        )
        self.chart_menu.pack(fill="x", pady=(0, 24))
        
        # Info box
        info_box = ctk.CTkFrame(controls_inner, fg_color="#EFF6FF", corner_radius=8)
        info_box.pack(fill="x", pady=(0, 24))
        
        info_inner = ctk.CTkFrame(info_box, fg_color="transparent")
        info_inner.pack(padx=14, pady=14)
        
        info_icon = ctk.CTkLabel(
            info_inner,
            text="ⓘ",
            font=("Segoe UI", 16),
            text_color="#2563EB"
        )
        info_icon.pack(side="left", padx=(0, 8))
        
        ctk.CTkLabel(
            info_inner, 
            text="Select a visualization type\nto explore your data.", 
            text_color="#1E40AF",
            font=("Segoe UI", 11),
            justify="left"
        ).pack(side="left", anchor="w")
        
        # Spacer
        ctk.CTkFrame(controls_inner, fg_color="transparent").pack(expand=True)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(controls_inner, mode="indeterminate", height=6)
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.pack_forget()
        
        self.status_label = ctk.CTkLabel(
            controls_inner,
            text="",
            text_color="#6B7280",
            font=("Segoe UI", 11)
        )
        self.status_label.pack(fill="x", pady=(0, 12))
        self.status_label.pack_forget()
        
        # Generate Button
        self.generate_btn = ctk.CTkButton(
            controls_inner,
            text="Generate Visualization",
            command=self.generate_viz,
            font=("Segoe UI", 14, "bold"),
            fg_color="#2563EB",
            hover_color="#1D4ED8",
            height=46,
            corner_radius=8
        )
        self.generate_btn.pack(fill="x")
        
        # Right Panel - Visualization
        self.viz_panel = ctk.CTkScrollableFrame(
            content, 
            fg_color="white", 
            corner_radius=12, 
            border_width=1, 
            border_color="#E5E7EB"
        )
        self.viz_panel.pack(side="right", fill="both", expand=True)
        
        # Initial message
        self.plot_placeholder()

    def plot_placeholder(self):
        """Show placeholder when no visualization is generated"""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        
        # Clear existing widgets
        for widget in self.viz_panel.winfo_children():
            widget.destroy()
            
        placeholder_container = ctk.CTkFrame(self.viz_panel, fg_color="transparent")
        placeholder_container.pack(expand=True, fill="both")
        
        placeholder_inner = ctk.CTkFrame(placeholder_container, fg_color="transparent")
        placeholder_inner.place(relx=0.5, rely=0.5, anchor="center")
        
        # Icon
        icon_bg = ctk.CTkFrame(placeholder_inner, fg_color="#F3F4F6", width=80, height=80, corner_radius=40)
        icon_bg.pack(pady=(0, 20))
        icon_bg.pack_propagate(False)
        
        icon_text = ctk.CTkLabel(icon_bg, text="◇", font=("Segoe UI", 36), text_color="#9CA3AF")
        icon_text.place(relx=0.5, rely=0.5, anchor="center")
        
        ctk.CTkLabel(
            placeholder_inner,
            text="No Visualization Generated",
            text_color="#111827",
            font=("Segoe UI", 18, "bold")
        ).pack(pady=(0, 8))
        
        ctk.CTkLabel(
            placeholder_inner,
            text="Load data and select a visualization type\nto see exploratory analysis charts",
            text_color="#6B7280",
            font=("Segoe UI", 13),
            justify="center"
        ).pack()

    def generate_viz(self):
        """Generate visualization based on selected type"""
        if self.is_generating:
            return
            
        df = self.app.get_dataframe()
        if df is None or df.empty:
            messagebox.showwarning("No Data", "Please load a dataset first from the Data Loader page.")
            return
        
        chart_type = self.chart_var.get()
        
        # Show progress
        self.is_generating = True
        self.progress_bar.pack(fill="x", pady=(0, 10))
        self.progress_bar.start()
        self.status_label.configure(text=f"Generating {chart_type}...")
        self.status_label.pack(fill="x", pady=(0, 12))
        self.generate_btn.configure(state="disabled", text="Generating...")
        
        # Run in thread
        thread = threading.Thread(target=self._generate_viz_thread, args=(df.copy(), chart_type))
        thread.daemon = True
        thread.start()
    
    def _generate_viz_thread(self, df, chart_type):
        """Generate visualization in background thread"""
        try:
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
        """Finish visualization generation"""
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
        """Handle visualization errors"""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.status_label.pack_forget()
        self.generate_btn.configure(state="normal", text="Generate Visualization")
        self.is_generating = False
        messagebox.showerror("Visualization Error", f"Error generating visualization:\n\n{error_msg}")
    
    def _display_plot(self, fig):
        """Display matplotlib figure"""
        if fig:
            self.canvas = FigureCanvasTkAgg(fig, master=self.viz_panel)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

    def show_statistics(self, df):
        """Show statistical summary"""
        stats_frame = ctk.CTkFrame(self.viz_panel, fg_color="transparent")
        stats_frame.pack(fill="both", expand=True, padx=24, pady=24)
        
        # Numeric summary
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            ctk.CTkLabel(
                stats_frame,
                text="Numeric Features Summary",
                font=("Segoe UI", 16, "bold"),
                text_color="#111827",
                anchor="w"
            ).pack(fill="x", pady=(0, 16))
            
            stats = numeric_df.describe()
            self.create_stats_table(stats_frame, stats)
        
        # Categorical summary
        cat_df = df.select_dtypes(include=['object'])
        if len(cat_df.columns) > 0:
            ctk.CTkLabel(
                stats_frame,
                text="Categorical Features Summary",
                font=("Segoe UI", 16, "bold"),
                text_color="#111827",
                anchor="w"
            ).pack(fill="x", pady=(24, 16))
            
            for col in cat_df.columns[:5]:
                value_counts = df[col].value_counts().head(5)
                self.create_value_count_card(stats_frame, col, value_counts)

    def create_stats_table(self, parent, stats):
        """Create a styled statistics table"""
        table_frame = ctk.CTkFrame(parent, fg_color="#F9FAFB", corner_radius=10, border_width=1, border_color="#E5E7EB")
        table_frame.pack(fill="x", pady=(0, 16))
        
        table_inner = ctk.CTkFrame(table_frame, fg_color="transparent")
        table_inner.pack(fill="x", padx=16, pady=16)
        
        # Headers
        headers = ["Statistic"] + list(stats.columns)
        for i, header in enumerate(headers):
            label = ctk.CTkLabel(
                table_inner,
                text=header if i == 0 else (header[:20] + "..." if len(header) > 20 else header),
                font=("Segoe UI", 11, "bold"),
                text_color="#374151",
                anchor="w"
            )
            label.grid(row=0, column=i, padx=12, pady=12, sticky="w")
        
        # Data rows
        for i, idx in enumerate(stats.index):
            # Row label
            ctk.CTkLabel(
                table_inner,
                text=idx,
                font=("Segoe UI", 11),
                text_color="#6B7280",
                anchor="w"
            ).grid(row=i+1, column=0, padx=12, pady=8, sticky="w")
            
            # Values
            for j, col in enumerate(stats.columns):
                val = stats.loc[idx, col]
                ctk.CTkLabel(
                    table_inner,
                    text=f"{val:.3f}" if not pd.isna(val) else "N/A",
                    font=("Segoe UI", 11),
                    text_color="#4B5563",
                    anchor="w"
                ).grid(row=i+1, column=j+1, padx=12, pady=8, sticky="w")

    def create_value_count_card(self, parent, column_name, value_counts):
        """Create a card showing value counts"""
        card = ctk.CTkFrame(parent, fg_color="#F9FAFB", corner_radius=10, border_width=1, border_color="#E5E7EB")
        card.pack(fill="x", pady=(0, 12))
        
        ctk.CTkLabel(
            card,
            text=column_name,
            font=("Segoe UI", 13, "bold"),
            text_color="#111827",
            anchor="w"
        ).pack(padx=18, pady=(14, 8), anchor="w")
        
        for val, count in value_counts.items():
            val_str = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
            ctk.CTkLabel(
                card,
                text=f"  •  {val_str}: {count}",
                font=("Segoe UI", 11),
                text_color="#6B7280",
                anchor="w"
            ).pack(padx=18, pady=3, anchor="w")
        
        ctk.CTkFrame(card, fg_color="transparent", height=8).pack()

    def _create_distributions(self, df):
        """Create distribution plots figure"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) == 0:
            raise ValueError("No numeric columns to visualize")
        
        n_cols = min(len(numeric_df.columns), 6)
        n_rows = (n_cols + 1) // 2
        
        fig = Figure(figsize=(14, 4.5*n_rows), dpi=100, facecolor='white')
        
        for i, col in enumerate(numeric_df.columns[:6]):
            ax = fig.add_subplot(n_rows, 2, i+1)
            data = numeric_df[col].dropna()
            ax.hist(data, bins=30, alpha=0.8, color='#2563EB', edgecolor='white', linewidth=1.2)
            ax.set_title(f'{col} Distribution', fontweight='bold', fontsize=12, pad=10)
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_facecolor('#FAFAFA')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        fig.tight_layout(pad=2.0)
        return fig

    def _create_correlation(self, df):
        """Create correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation")
        
        corr = numeric_df.corr()
        
        fig = Figure(figsize=(max(10, len(corr.columns)*0.8), max(8, len(corr.columns)*0.7)), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        im = ax.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(corr.columns, fontsize=10)
        
        # Add correlation values
        for i in range(len(corr)):
            for j in range(len(corr)):
                val = corr.iloc[i, j]
                text_color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha="center", va="center", 
                       color=text_color, fontsize=9, fontweight='bold')
        
        ax.set_title('Correlation Heatmap', fontweight='bold', fontsize=14, pad=15)
        cbar = fig.colorbar(im, ax=ax, label='Correlation Coefficient')
        cbar.ax.tick_params(labelsize=9)
        fig.tight_layout()
        
        return fig

    def _create_boxplots(self, df):
        """Create box plots for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) == 0:
            raise ValueError("No numeric columns to visualize")
        
        n_cols = min(len(numeric_df.columns), 6)
        n_rows = (n_cols + 1) // 2
        
        fig = Figure(figsize=(14, 4.5*n_rows), dpi=100, facecolor='white')
        
        for i, col in enumerate(numeric_df.columns[:6]):
            ax = fig.add_subplot(n_rows, 2, i+1)
            data = numeric_df[col].dropna()
            bp = ax.boxplot(data, vert=True, patch_artist=True, 
                           boxprops=dict(facecolor='#2563EB', alpha=0.7),
                           medianprops=dict(color='#1E40AF', linewidth=2),
                           whiskerprops=dict(color='#2563EB'),
                           capprops=dict(color='#2563EB'),
                           flierprops=dict(marker='o', markerfacecolor='#EF4444', 
                                         markersize=4, alpha=0.5))
            
            ax.set_title(f'{col} Box Plot', fontweight='bold', fontsize=12, pad=10)
            ax.set_ylabel(col, fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.5)
            ax.set_facecolor('#FAFAFA')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        fig.tight_layout(pad=2.0)
        return fig

    def _create_scatter_matrix(self, df):
        """Create scatter matrix for first few numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            raise ValueError("Need at least 2 numeric columns")
        
        cols_to_use = numeric_df.columns[:4]
        n = len(cols_to_use)
        
        fig = Figure(figsize=(14, 14), dpi=100, facecolor='white')
        
        for i in range(n):
            for j in range(n):
                ax = fig.add_subplot(n, n, i*n + j + 1)
                if i == j:
                    # Diagonal: histogram
                    data = numeric_df[cols_to_use[i]].dropna()
                    ax.hist(data, bins=20, alpha=0.8, color='#2563EB', edgecolor='white')
                else:
                    # Off-diagonal: scatter
                    ax.scatter(numeric_df[cols_to_use[j]], numeric_df[cols_to_use[i]], 
                             alpha=0.4, s=15, color='#2563EB')
                
                if i == n-1:
                    ax.set_xlabel(cols_to_use[j], fontsize=10)
                else:
                    ax.set_xticklabels([])
                
                if j == 0:
                    ax.set_ylabel(cols_to_use[i], fontsize=10)
                else:
                    ax.set_yticklabels([])
                
                ax.grid(True, alpha=0.2, linewidth=0.5)
                ax.set_facecolor('#FAFAFA')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        fig.suptitle('Scatter Matrix', fontweight='bold', fontsize=15, y=0.995)
        fig.tight_layout(pad=1.5)
        
        return fig

    def _create_missing_data(self, df):
        """Visualize missing data"""
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            raise ValueError("No missing values found in the dataset")
        
        fig = Figure(figsize=(12, max(6, len(missing)*0.5)), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        y_pos = range(len(missing))
        bars = ax.barh(y_pos, missing.values, color='#EF4444', alpha=0.8, height=0.6)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, missing.values)):
            ax.text(val + max(missing.values)*0.02, i, str(val), 
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(missing.index, fontsize=11)
        ax.set_xlabel('Number of Missing Values', fontsize=12, fontweight='bold')
        ax.set_title('Missing Data by Column', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x', linewidth=0.5)
        ax.set_facecolor('#FAFAFA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        return fig

    def _create_pairwise_corr(self, df):
        """Show pairwise correlations as bar chart"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            raise ValueError("Need at least 2 numeric columns")
        
        corr = numeric_df.corr()
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_pairs.append((f"{corr.columns[i]} vs {corr.columns[j]}", corr.iloc[i, j]))
        
        corr_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        corr_pairs = corr_pairs[:15]
        
        labels, values = zip(*corr_pairs)
        
        fig = Figure(figsize=(12, max(6, len(labels)*0.5)), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        colors = ['#10B981' if v > 0 else '#EF4444' for v in values]
        y_pos = range(len(labels))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8, height=0.6)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            x_pos = val + (0.05 if val > 0 else -0.05)
            ax.text(x_pos, i, f'{val:.3f}', va='center', ha='left' if val > 0 else 'right',
                   fontsize=9, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.set_title('Top Pairwise Correlations', fontsize=14, fontweight='bold', pad=15)
        ax.axvline(x=0, color='#374151', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x', linewidth=0.5)
        ax.set_facecolor('#FAFAFA')
        ax.set_xlim(-1.1, 1.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        return fig