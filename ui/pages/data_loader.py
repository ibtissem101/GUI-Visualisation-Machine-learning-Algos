import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os
import numpy as np
import threading

# Common placeholder values that represent missing/empty data
EMPTY_VALUE_PLACEHOLDERS = {'', '?', 'N/A', 'n/a', 'NA', 'na', 'NULL', 'null', 'None', 'none', 
                            '-', '--', 'NaN', 'nan', '.', 'missing', 'undefined'}

class DropZone(ctk.CTkFrame):
    def __init__(self, master, load_callback):
        super().__init__(master)
        self.load_callback = load_callback
        self.configure(fg_color="#FAFAFA", border_width=2, border_color="#D1D5DB", corner_radius=12)
        self.pack_propagate(False)
        self.configure(height=280)
        self.setup_ui()
        
    def setup_ui(self):
        self.inner_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.inner_frame.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(self.inner_frame, text="📁", font=("Segoe UI", 56)).pack(pady=(0, 20))
        ctk.CTkLabel(self.inner_frame, text="Load Dataset", font=("Segoe UI", 22, "bold"), text_color="#111827").pack(pady=(0, 10))
        ctk.CTkLabel(self.inner_frame, text="Drag & Drop your file here or click to browse\nSupported formats: CSV, JSON, XLSX", font=("Segoe UI", 14), text_color="#6B7280", justify="center").pack(pady=(0, 25))
        self.browse_btn = ctk.CTkButton(self.inner_frame, text="Browse Files", command=self.browse_file, font=("Segoe UI", 14, "bold"), fg_color="#2563EB", hover_color="#1D4ED8", height=48, width=180, corner_radius=8)
        self.browse_btn.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Data Files", "*.csv *.xlsx *.json"), ("All Files", "*.*")])
        if file_path:
            self.load_callback(file_path)

class TableFrame(ctk.CTkFrame):
    def __init__(self, master, df=None):
        super().__init__(master, fg_color="white", corner_radius=12, border_width=1, border_color="#E5E7EB")
        self.df = df
        self.canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self.v_scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.h_scrollbar = ctk.CTkScrollbar(self, orientation="horizontal", command=self.canvas.xview)
        self.scrollable_frame = ctk.CTkFrame(self.canvas, fg_color="white")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        if df is not None:
            self.populate_table(df)
        else:
            self.populate_initial()

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def populate_initial(self):
        columns = ["CUSTOMER ID", "AGE", "GENDER", "PLAN", "TENURE"]
        data = [["CUST-001", "34", "Male", "Premium", "24 months"], ["CUST-002", "28", "Female", "Basic", "12 months"]]
        self.create_table_grid(columns, data)

    def populate_table(self, df):
        df_head = df.head(100)
        if df_head.shape[1] > 15:
            df_head = df_head.iloc[:, :15]
        columns = [str(c).upper()[:25] for c in df_head.columns]
        data = df_head.astype(str).values.tolist()
        self.create_table_grid(columns, data)

    def create_table_grid(self, columns, data):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        for i, col in enumerate(columns):
            ctk.CTkLabel(self.scrollable_frame, text=col, font=("Segoe UI", 11, "bold"), text_color="#374151", fg_color="#F9FAFB", padx=12, pady=10, anchor="w", width=120).grid(row=0, column=i, sticky="ew", padx=(0, 1), pady=(0, 1))
        for r, row in enumerate(data):
            for c, val in enumerate(row):
                if val == "nan": val = ""
                val = val[:30] + "..." if len(val) > 30 else val
                bg = "white" if r % 2 == 0 else "#FAFAFA"
                ctk.CTkLabel(self.scrollable_frame, text=val, font=("Segoe UI", 11), text_color="#4B5563", fg_color=bg, padx=12, pady=8, anchor="w", width=120).grid(row=r+1, column=c, sticky="ew", padx=(0, 1), pady=(0, 1))


def count_empty_values_fast(df):
    """Fast empty value counting using vectorized operations"""
    empty_counts = {}
    total_rows = len(df)
    
    for col in df.columns:
        # Count nulls
        null_count = int(df[col].isnull().sum())
        placeholder_count = 0
        
        # Only check string columns for placeholders
        if df[col].dtype == 'object':
            # Vectorized: convert to string, strip, check if in set
            str_col = df[col].dropna().astype(str).str.strip()
            placeholder_count = int(str_col.isin(EMPTY_VALUE_PLACEHOLDERS).sum())
        
        total = null_count + placeholder_count
        if total > 0:
            empty_counts[col] = total
    
    return dict(sorted(empty_counts.items(), key=lambda x: x[1], reverse=True))


class DataLoaderPage(ctk.CTkFrame):
    def __init__(self, master, app_instance):
        super().__init__(master, fg_color="#F9FAFB", corner_radius=0)
        self.app = app_instance
        self.preprocessing_options = {}
        self.missing_details_expanded = True
        self.original_stats = None  # Store original stats for comparison
        self.setup_ui()
        self.after(100, self._restore_state)
    
    def _restore_state(self):
        df = self.app.get_dataframe()
        if df is not None and self.app.file_path:
            self.update_ui(df, self.app.file_path)
        
    def setup_ui(self):
        ctk.CTkLabel(self, text="Data Loading & Preprocessing", font=("Segoe UI", 26, "bold"), text_color="#0F172A", anchor="w").pack(padx=30, pady=(30, 5), anchor="w")
        ctk.CTkLabel(self, text="Load your dataset, preview data, and apply preprocessing steps before analysis.", font=("Segoe UI", 14), text_color="#64748B", anchor="w").pack(padx=30, pady=(0, 20), anchor="w")

        self.view_var = ctk.StringVar(value="Load Data")
        self.view_switcher = ctk.CTkSegmentedButton(self, values=["Load Data", "Preview", "Preprocessing"], variable=self.view_var, command=self.switch_view, font=("Segoe UI", 12, "bold"), height=32)
        self.view_switcher.pack(padx=30, pady=(0, 20), anchor="w")
        
        self.content_area = ctk.CTkFrame(self, fg_color="transparent")
        self.content_area.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        
        self.load_frame = ctk.CTkFrame(self.content_area, fg_color="white", corner_radius=8, border_width=1, border_color="#E2E8F0")
        self.create_load_view()
        
        self.preview_frame = ctk.CTkFrame(self.content_area, fg_color="white", corner_radius=8, border_width=1, border_color="#E2E8F0")
        self.create_preview_view()
        
        self.preprocess_frame = ctk.CTkFrame(self.content_area, fg_color="white", corner_radius=8, border_width=1, border_color="#E2E8F0")
        self.create_preprocessing_view()
        
        self.switch_view("Load Data")

    def switch_view(self, view_name):
        self.load_frame.pack_forget()
        self.preview_frame.pack_forget()
        self.preprocess_frame.pack_forget()
        
        if view_name == "Load Data":
            self.load_frame.pack(fill="both", expand=True)
        elif view_name == "Preview":
            self.preview_frame.pack(fill="both", expand=True)
            df = self.app.get_dataframe()
            if df is not None:
                self.table_frame.populate_table(df)
        elif view_name == "Preprocessing":
            self.preprocess_frame.pack(fill="both", expand=True)

    def create_load_view(self):
        inner = ctk.CTkFrame(self.load_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=40, pady=40)
        
        self.file_input_container = ctk.CTkFrame(inner, fg_color="transparent")
        self.file_input_container.pack(fill="x", pady=(0, 30))
        
        self.drop_zone = DropZone(self.file_input_container, self.load_file)
        self.drop_zone.pack(fill="x")
        
        self.file_widget = ctk.CTkFrame(self.file_input_container, fg_color="#F0FDF4", corner_radius=10, border_width=1, border_color="#BBF7D0")
        file_inner = ctk.CTkFrame(self.file_widget, fg_color="transparent")
        file_inner.pack(fill="x", padx=20, pady=16)
        ctk.CTkLabel(file_inner, text="✓", text_color="#16A34A", font=("Segoe UI", 20, "bold")).pack(side="left", padx=(0, 12))
        file_info = ctk.CTkFrame(file_inner, fg_color="transparent")
        file_info.pack(side="left", fill="x", expand=True)
        self.file_label = ctk.CTkLabel(file_info, text="...", text_color="#111827", font=("Segoe UI", 14, "bold"), anchor="w")
        self.file_label.pack(anchor="w")
        self.file_size_label = ctk.CTkLabel(file_info, text="", text_color="#6B7280", font=("Segoe UI", 11), anchor="w")
        self.file_size_label.pack(anchor="w")
        ctk.CTkButton(file_inner, text="×", width=32, height=32, fg_color="#FEE2E2", text_color="#DC2626", hover_color="#FECACA", font=("Segoe UI", 20), corner_radius=6, command=self.clear_file).pack(side="right")
        
        # Dataset Overview
        self.overview_container = ctk.CTkFrame(inner, fg_color="#F9FAFB", corner_radius=10, border_width=1, border_color="#E5E7EB")
        self.overview_container.pack(fill="x", pady=(0, 20))
        overview_inner = ctk.CTkFrame(self.overview_container, fg_color="transparent")
        overview_inner.pack(fill="x", padx=20, pady=16)
        
        # Header with preprocessing indicator
        header_row = ctk.CTkFrame(overview_inner, fg_color="transparent")
        header_row.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(header_row, text="📈 Dataset Overview", font=("Segoe UI", 16, "bold"), text_color="#111827", anchor="w").pack(side="left")
        self.preprocessing_badge = ctk.CTkLabel(header_row, text="", font=("Segoe UI", 11), text_color="#059669", anchor="e")
        self.preprocessing_badge.pack(side="right")
        
        stats_frame = ctk.CTkFrame(overview_inner, fg_color="transparent")
        stats_frame.pack(fill="x")
        self.rows_card = self.create_stat_card(stats_frame, "📊 Rows", "0")
        self.rows_card.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.cols_card = self.create_stat_card(stats_frame, "📋 Columns", "0")
        self.cols_card.pack(side="left", fill="x", expand=True, padx=(10, 10))
        self.missing_card = self.create_stat_card(stats_frame, "⚠️ Empty", "0")
        self.missing_card.pack(side="left", fill="x", expand=True, padx=(10, 10))
        self.duplicates_card = self.create_stat_card(stats_frame, "🔄 Duplicates", "0")
        self.duplicates_card.pack(side="left", fill="x", expand=True, padx=(10, 0))
        
        # Empty Values by Column
        self.missing_by_column_frame = ctk.CTkFrame(overview_inner, fg_color="#FEF9C3", corner_radius=8, border_width=1, border_color="#FDE047")
        self.missing_by_column_frame.pack(fill="x", pady=(16, 0))
        missing_header = ctk.CTkFrame(self.missing_by_column_frame, fg_color="transparent")
        missing_header.pack(fill="x", padx=12, pady=(10, 0))
        self.missing_toggle_btn = ctk.CTkButton(missing_header, text="▼ Empty Values by Column", font=("Segoe UI", 12, "bold"), text_color="#92400E", fg_color="transparent", hover_color="#FEF08A", anchor="w", height=24, command=self.toggle_missing_details)
        self.missing_toggle_btn.pack(side="left", fill="x", expand=True)
        self.missing_details_frame = ctk.CTkFrame(self.missing_by_column_frame, fg_color="transparent")
        self.missing_details_frame.pack(fill="x", padx=12, pady=(6, 10))
        self.missing_scroll_frame = ctk.CTkScrollableFrame(self.missing_details_frame, fg_color="#FFFBEB", corner_radius=6, height=100)
        self.missing_scroll_frame.pack(fill="x")
        ctk.CTkLabel(self.missing_scroll_frame, text="📊 Load a dataset to see empty values analysis", font=("Segoe UI", 12), text_color="#78716C", anchor="w").pack(fill="x", padx=12, pady=12)
        
        # Status
        self.status_frame = ctk.CTkFrame(inner, fg_color="#FEF3C7", corner_radius=8)
        self.status_frame.pack(fill="x")
        self.status_label = ctk.CTkLabel(self.status_frame, text="📁 No data loaded yet. Click 'Browse Files' to get started.", fg_color="transparent", text_color="#92400E", font=("Segoe UI", 13), anchor="w")
        self.status_label.pack(fill="x", padx=16, pady=14)

    def toggle_missing_details(self):
        if self.missing_details_expanded:
            self.missing_details_frame.pack_forget()
            self.missing_toggle_btn.configure(text="▶ Empty Values by Column")
            self.missing_details_expanded = False
        else:
            self.missing_details_frame.pack(fill="x", padx=12, pady=(6, 10))
            self.missing_toggle_btn.configure(text="▼ Empty Values by Column")
            self.missing_details_expanded = True

    def update_missing_by_column(self, df, empty_per_col):
        for widget in self.missing_scroll_frame.winfo_children():
            widget.destroy()
        
        if df is None:
            ctk.CTkLabel(self.missing_scroll_frame, text="📊 Load a dataset to see empty values analysis", font=("Segoe UI", 12), text_color="#78716C", anchor="w").pack(fill="x", padx=12, pady=12)
            return
        
        if len(empty_per_col) == 0:
            ctk.CTkLabel(self.missing_scroll_frame, text="✅ No empty values detected!", font=("Segoe UI", 12), text_color="#059669", anchor="w").pack(fill="x", padx=12, pady=12)
            return
        
        # Header
        hdr = ctk.CTkFrame(self.missing_scroll_frame, fg_color="transparent")
        hdr.pack(fill="x", padx=8, pady=(8, 4))
        ctk.CTkLabel(hdr, text="Column", font=("Segoe UI", 11, "bold"), text_color="#78716C", width=180, anchor="w").pack(side="left")
        ctk.CTkLabel(hdr, text="Count", font=("Segoe UI", 11, "bold"), text_color="#78716C", width=80, anchor="center").pack(side="left", padx=(10, 0))
        ctk.CTkLabel(hdr, text="%", font=("Segoe UI", 11, "bold"), text_color="#78716C", width=60, anchor="center").pack(side="left", padx=(10, 0))
        ctk.CTkFrame(self.missing_scroll_frame, fg_color="#E5E7EB", height=1).pack(fill="x", padx=8, pady=4)
        
        total_rows = len(df)
        total_empty = 0
        for col_name, empty_count in list(empty_per_col.items())[:10]:  # Show top 10
            total_empty += empty_count
            row = ctk.CTkFrame(self.missing_scroll_frame, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=1)
            display_name = str(col_name)[:20] + "..." if len(str(col_name)) > 20 else str(col_name)
            ctk.CTkLabel(row, text=display_name, font=("Segoe UI", 10), text_color="#374151", width=180, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=f"{empty_count:,}", font=("Segoe UI", 10, "bold"), text_color="#DC2626", width=80, anchor="center").pack(side="left", padx=(10, 0))
            pct = (empty_count / total_rows) * 100
            pct_color = "#DC2626" if pct > 50 else "#F59E0B" if pct > 20 else "#059669"
            ctk.CTkLabel(row, text=f"{pct:.1f}%", font=("Segoe UI", 10), text_color=pct_color, width=60, anchor="center").pack(side="left", padx=(10, 0))
        
        if len(empty_per_col) > 10:
            ctk.CTkLabel(self.missing_scroll_frame, text=f"... and {len(empty_per_col) - 10} more columns", font=("Segoe UI", 10), text_color="#78716C", anchor="w").pack(fill="x", padx=8, pady=4)

    def create_preview_view(self):
        inner = ctk.CTkFrame(self.preview_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=20, pady=20)
        header = ctk.CTkFrame(inner, fg_color="transparent")
        header.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(header, text="Data Preview", font=("Segoe UI", 18, "bold"), text_color="#111827", anchor="w").pack(side="left")
        self.preview_info = ctk.CTkLabel(header, text="Showing first 100 rows", font=("Segoe UI", 12), text_color="#6B7280", anchor="e")
        self.preview_info.pack(side="right")
        self.table_frame = TableFrame(inner)
        self.table_frame.pack(fill="both", expand=True)

    def create_preprocessing_view(self):
        inner = ctk.CTkFrame(self.preprocess_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=40, pady=40)
        ctk.CTkLabel(inner, text="Preprocessing Options", font=("Segoe UI", 18, "bold"), text_color="#111827", anchor="w").pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(inner, text="Select preprocessing steps to apply to your dataset", font=("Segoe UI", 13), text_color="#6B7280", anchor="w").pack(fill="x", pady=(0, 24))
        
        options_frame = ctk.CTkFrame(inner, fg_color="transparent")
        options_frame.pack(fill="x", pady=(0, 30))
        options_frame.grid_columnconfigure(0, weight=1)
        options_frame.grid_columnconfigure(1, weight=1)
        
        self.create_preprocess_option(options_frame, "Missing Values", ["None", "Remove Rows", "Mean", "Median", "Mode", "Forward", "Backward"], 0, 0)
        self.create_preprocess_option(options_frame, "Duplicates", ["None", "Remove All", "Keep First", "Keep Last"], 0, 1)
        self.create_preprocess_option(options_frame, "Normalization", ["None", "Min-Max", "Z-Score", "Robust"], 1, 0)
        self.create_preprocess_option(options_frame, "Outliers", ["None", "Remove IQR", "Cap IQR", "Z-Score >3"], 1, 1)
        self.create_preprocess_option(options_frame, "Encoding", ["None", "One-Hot", "Label"], 2, 0)
        
        self.preprocess_progress = ctk.CTkProgressBar(inner, mode="indeterminate", height=4)
        self.preprocess_status = ctk.CTkLabel(inner, text="", text_color="#6B7280", font=("Segoe UI", 12))
        
        btn_frame = ctk.CTkFrame(inner, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(0, 20))
        self.apply_btn = ctk.CTkButton(btn_frame, text="Apply Preprocessing", command=self.apply_preprocessing, font=("Segoe UI", 14, "bold"), fg_color="#059669", hover_color="#047857", height=48, width=220, corner_radius=8)
        self.apply_btn.pack(side="left")
        
        steps_frame = ctk.CTkFrame(inner, fg_color="transparent")
        steps_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(steps_frame, text="Applied Steps", font=("Segoe UI", 16, "bold"), text_color="#111827", anchor="w").pack(fill="x", pady=(0, 12))
        self.steps_container = ctk.CTkFrame(steps_frame, fg_color="#F9FAFB", corner_radius=8)
        self.steps_container.pack(fill="both", expand=True)
        self.steps_label = ctk.CTkLabel(self.steps_container, text="No preprocessing steps applied yet.", text_color="#6B7280", font=("Segoe UI", 13), anchor="w")
        self.steps_label.pack(fill="x", padx=16, pady=16)

    def create_preprocess_option(self, parent, title, options, row, col):
        card = ctk.CTkFrame(parent, fg_color="#F9FAFB", corner_radius=8)
        card.grid(row=row, column=col, padx=8, pady=8, sticky="ew")
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=14)
        ctk.CTkLabel(inner, text=title, font=("Segoe UI", 13, "bold"), text_color="#374151", anchor="w").pack(fill="x", pady=(0, 8))
        var = ctk.StringVar(value=options[0])
        ctk.CTkOptionMenu(inner, values=options, variable=var, fg_color="#2563EB", button_color="#2563EB", button_hover_color="#1D4ED8", dropdown_fg_color="white", dropdown_hover_color="#F3F4F6", dropdown_text_color="#111827", font=("Segoe UI", 12), height=36, corner_radius=6).pack(fill="x")
        self.preprocessing_options[title] = var

    def create_stat_card(self, parent, title, value):
        card = ctk.CTkFrame(parent, fg_color="white", corner_radius=8, border_width=1, border_color="#E5E7EB")
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=14, pady=12)
        ctk.CTkLabel(inner, text=title, text_color="#6B7280", font=("Segoe UI", 11), anchor="w").pack(anchor="w", pady=(0, 2))
        val_label = ctk.CTkLabel(inner, text=value, text_color="#111827", font=("Segoe UI", 22, "bold"), anchor="w")
        val_label.pack(anchor="w")
        # Add change indicator
        change_label = ctk.CTkLabel(inner, text="", text_color="#059669", font=("Segoe UI", 9), anchor="w")
        change_label.pack(anchor="w")
        card.value_label = val_label
        card.change_label = change_label
        return card

    def load_file(self, file_path):
        self.drop_zone.configure(fg_color="#EFF6FF", border_color="#BFDBFE")
        self.drop_zone.browse_btn.configure(state="disabled", text="Loading...")
        thread = threading.Thread(target=self._load_file_thread, args=(file_path,))
        thread.daemon = True
        thread.start()
    
    def _load_file_thread(self, file_path):
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            self.after(0, lambda: self._finish_load(df, file_path))
        except Exception as e:
            self.after(0, lambda err=str(e): self._handle_load_error(err))
    
    def _finish_load(self, df, file_path):
        self.drop_zone.configure(fg_color="#FAFAFA", border_color="#D1D5DB")
        self.drop_zone.browse_btn.configure(state="normal", text="Browse Files")
        self.app.set_dataframe(df, file_path)
        
        # Store original stats
        empty_per_col = count_empty_values_fast(df)
        self.original_stats = {
            'rows': df.shape[0],
            'cols': df.shape[1],
            'empty': sum(empty_per_col.values()),
            'duplicates': df.duplicated().sum()
        }
        self.preprocessing_badge.configure(text="")
        
        self.update_ui(df, file_path, empty_per_col, show_changes=False)
    
    def _handle_load_error(self, error_msg):
        self.drop_zone.configure(fg_color="#FAFAFA", border_color="#D1D5DB")
        self.drop_zone.browse_btn.configure(state="normal", text="Browse Files")
        messagebox.showerror("Load Error", f"Error loading file: {error_msg}")

    def update_ui(self, df, file_path, empty_per_col=None, show_changes=False):
        if df is None: return
        
        if empty_per_col is None:
            empty_per_col = count_empty_values_fast(df)
        
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
        
        self.file_label.configure(text=filename)
        self.file_size_label.configure(text=f"{size_str} • {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        self.drop_zone.pack_forget()
        self.file_widget.pack(fill="x")
        
        total_empty = sum(empty_per_col.values())
        duplicates = df.duplicated().sum()
        
        self.rows_card.value_label.configure(text=f"{df.shape[0]:,}")
        self.cols_card.value_label.configure(text=str(df.shape[1]))
        self.missing_card.value_label.configure(text=str(total_empty))
        self.duplicates_card.value_label.configure(text=str(duplicates))
        
        # Show changes from original if preprocessing was applied
        if show_changes and self.original_stats:
            orig = self.original_stats
            
            # Rows change
            row_diff = df.shape[0] - orig['rows']
            if row_diff != 0:
                self.rows_card.change_label.configure(text=f"({row_diff:+,})", text_color="#DC2626" if row_diff < 0 else "#059669")
            else:
                self.rows_card.change_label.configure(text="")
            
            # Cols change
            col_diff = df.shape[1] - orig['cols']
            if col_diff != 0:
                self.cols_card.change_label.configure(text=f"({col_diff:+})", text_color="#DC2626" if col_diff < 0 else "#059669")
            else:
                self.cols_card.change_label.configure(text="")
            
            # Empty change
            empty_diff = total_empty - orig['empty']
            if empty_diff != 0:
                self.missing_card.change_label.configure(text=f"({empty_diff:+,})", text_color="#059669" if empty_diff < 0 else "#DC2626")
            else:
                self.missing_card.change_label.configure(text="")
            
            # Duplicates change
            dup_diff = duplicates - orig['duplicates']
            if dup_diff != 0:
                self.duplicates_card.change_label.configure(text=f"({dup_diff:+})", text_color="#059669" if dup_diff < 0 else "#DC2626")
            else:
                self.duplicates_card.change_label.configure(text="")
        else:
            # Clear change indicators
            self.rows_card.change_label.configure(text="")
            self.cols_card.change_label.configure(text="")
            self.missing_card.change_label.configure(text="")
            self.duplicates_card.change_label.configure(text="")
        
        self.update_missing_by_column(df, empty_per_col)
        
        self.table_frame.populate_table(df)
        self.preview_info.configure(text=f"Showing first {min(100, df.shape[0])} rows of {df.shape[0]:,}")
        
        self.status_frame.configure(fg_color="#ECFDF5")
        self.status_label.configure(text=f"✓ Loaded '{filename}' successfully. Switch to 'Preview' tab to see the data.", text_color="#059669")

    def clear_file(self):
        self.app.set_dataframe(None, None)
        self.file_widget.pack_forget()
        self.drop_zone.pack(fill="x")
        
        self.original_stats = None
        self.preprocessing_badge.configure(text="")
        
        self.rows_card.value_label.configure(text="0")
        self.cols_card.value_label.configure(text="0")
        self.missing_card.value_label.configure(text="0")
        self.duplicates_card.value_label.configure(text="0")
        
        self.rows_card.change_label.configure(text="")
        self.cols_card.change_label.configure(text="")
        self.missing_card.change_label.configure(text="")
        self.duplicates_card.change_label.configure(text="")
        
        self.update_missing_by_column(None, {})
        
        self.table_frame.populate_initial()
        self.preview_info.configure(text="Showing first 100 rows")
        
        self.status_frame.configure(fg_color="#FEF3C7")
        self.status_label.configure(text="📁 No data loaded yet. Click 'Browse Files' to get started.", text_color="#92400E")
        self.steps_label.configure(text="No preprocessing steps applied yet.")

    def apply_preprocessing(self):
        df = self.app.get_dataframe()
        if df is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self.preprocess_progress.pack(fill="x", pady=(0, 8))
        self.preprocess_progress.start()
        self.preprocess_status.configure(text="Applying preprocessing...")
        self.preprocess_status.pack(fill="x", pady=(0, 8))
        self.apply_btn.configure(state="disabled", text="Processing...")
        
        thread = threading.Thread(target=self._apply_preprocessing_thread, args=(df.copy(),))
        thread.daemon = True
        thread.start()
    
    def _apply_preprocessing_thread(self, df):
        df_processed = df.copy()
        steps_applied = []
        
        try:
            # First, replace placeholder values with NaN for all object columns
            for col in df_processed.columns:
                if df_processed[col].dtype == 'object':
                    mask = df_processed[col].astype(str).str.strip().isin(EMPTY_VALUE_PLACEHOLDERS)
                    df_processed.loc[mask, col] = np.nan
            
            missing_method = self.preprocessing_options["Missing Values"].get()
            if missing_method != "None":
                before_rows = len(df_processed)
                if missing_method == "Remove Rows":
                    df_processed = df_processed.dropna()
                    steps_applied.append(f"✓ Removed {before_rows - len(df_processed)} rows with missing values")
                elif missing_method == "Mean":
                    num_cols = df_processed.select_dtypes(include=[np.number]).columns
                    df_processed[num_cols] = df_processed[num_cols].fillna(df_processed[num_cols].mean())
                    steps_applied.append("✓ Filled numeric missing values with mean")
                elif missing_method == "Median":
                    num_cols = df_processed.select_dtypes(include=[np.number]).columns
                    df_processed[num_cols] = df_processed[num_cols].fillna(df_processed[num_cols].median())
                    steps_applied.append("✓ Filled numeric missing values with median")
                elif missing_method == "Mode":
                    for col in df_processed.columns:
                        mode_val = df_processed[col].mode()
                        if len(mode_val) > 0:
                            df_processed[col] = df_processed[col].fillna(mode_val.iloc[0])
                    steps_applied.append("✓ Filled missing values with mode")
                elif missing_method == "Forward":
                    df_processed = df_processed.ffill()
                    steps_applied.append("✓ Forward filled missing values")
                elif missing_method == "Backward":
                    df_processed = df_processed.bfill()
                    steps_applied.append("✓ Backward filled missing values")
            
            dup_method = self.preprocessing_options["Duplicates"].get()
            if dup_method != "None":
                before_rows = len(df_processed)
                if dup_method == "Remove All":
                    df_processed = df_processed.drop_duplicates(keep=False)
                elif dup_method == "Keep First":
                    df_processed = df_processed.drop_duplicates(keep='first')
                elif dup_method == "Keep Last":
                    df_processed = df_processed.drop_duplicates(keep='last')
                removed = before_rows - len(df_processed)
                if removed > 0:
                    steps_applied.append(f"✓ Removed {removed} duplicate rows")
            
            norm_method = self.preprocessing_options["Normalization"].get()
            if norm_method != "None":
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    if norm_method == "Min-Max":
                        for col in numeric_cols:
                            min_val, max_val = df_processed[col].min(), df_processed[col].max()
                            if max_val != min_val:
                                df_processed[col] = (df_processed[col] - min_val) / (max_val - min_val)
                        steps_applied.append("✓ Applied Min-Max normalization")
                    elif norm_method == "Z-Score":
                        for col in numeric_cols:
                            mean_val, std_val = df_processed[col].mean(), df_processed[col].std()
                            if std_val != 0:
                                df_processed[col] = (df_processed[col] - mean_val) / std_val
                        steps_applied.append("✓ Applied Z-Score normalization")
                    elif norm_method == "Robust":
                        for col in numeric_cols:
                            median_val = df_processed[col].median()
                            q1, q3 = df_processed[col].quantile(0.25), df_processed[col].quantile(0.75)
                            iqr = q3 - q1
                            if iqr != 0:
                                df_processed[col] = (df_processed[col] - median_val) / iqr
                        steps_applied.append("✓ Applied Robust scaling")
            
            outlier_method = self.preprocessing_options.get("Outliers", ctk.StringVar(value="None")).get()
            if outlier_method != "None":
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    before_rows = len(df_processed)
                    if outlier_method == "Remove IQR":
                        for col in numeric_cols:
                            Q1, Q3 = df_processed[col].quantile(0.25), df_processed[col].quantile(0.75)
                            IQR = Q3 - Q1
                            df_processed = df_processed[(df_processed[col] >= Q1 - 1.5*IQR) & (df_processed[col] <= Q3 + 1.5*IQR)]
                        steps_applied.append(f"✓ Removed {before_rows - len(df_processed)} outlier rows (IQR)")
                    elif outlier_method == "Cap IQR":
                        for col in numeric_cols:
                            Q1, Q3 = df_processed[col].quantile(0.25), df_processed[col].quantile(0.75)
                            IQR = Q3 - Q1
                            df_processed[col] = df_processed[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
                        steps_applied.append("✓ Capped outliers using IQR")
                    elif outlier_method == "Z-Score >3":
                        for col in numeric_cols:
                            mean_val, std_val = df_processed[col].mean(), df_processed[col].std()
                            if std_val != 0:
                                z = np.abs((df_processed[col] - mean_val) / std_val)
                                df_processed = df_processed[z < 3]
                        steps_applied.append(f"✓ Removed {before_rows - len(df_processed)} outlier rows (Z>3)")
            
            encoding_method = self.preprocessing_options["Encoding"].get()
            if encoding_method != "None":
                cat_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
                if len(cat_cols) > 0:
                    if encoding_method == "One-Hot":
                        df_processed = pd.get_dummies(df_processed, columns=cat_cols)
                        steps_applied.append(f"✓ One-Hot encoded {len(cat_cols)} columns")
                    elif encoding_method == "Label":
                        for col in cat_cols:
                            df_processed[col] = pd.factorize(df_processed[col])[0]
                        steps_applied.append(f"✓ Label encoded {len(cat_cols)} columns")
            
            self.after(0, lambda: self._finish_preprocessing(df_processed, steps_applied))
        except Exception as e:
            self.after(0, lambda err=str(e): self._handle_preprocessing_error(err))
    
    def _finish_preprocessing(self, df_processed, steps_applied):
        self.preprocess_progress.stop()
        self.preprocess_progress.pack_forget()
        self.preprocess_status.configure(text="✓ Preprocessing complete!")
        self.after(2000, lambda: self.preprocess_status.pack_forget())
        self.apply_btn.configure(state="normal", text="Apply Preprocessing")
        
        # Update the dataframe
        self.app.set_dataframe(df_processed, self.app.file_path)
        
        # Update UI with changes shown
        empty_per_col = count_empty_values_fast(df_processed)
        self.update_ui(df_processed, self.app.file_path, empty_per_col, show_changes=True)
        
        # Show preprocessing badge
        self.preprocessing_badge.configure(text="✓ Preprocessed")
        
        if steps_applied:
            self.steps_label.configure(text="\n".join(steps_applied), text_color="#059669")
            messagebox.showinfo("Success", f"Applied {len(steps_applied)} preprocessing step(s)!\n\nCheck Dataset Overview for changes.")
        else:
            messagebox.showinfo("No Changes", "No preprocessing options were selected.")
    
    def _handle_preprocessing_error(self, error_msg):
        self.preprocess_progress.stop()
        self.preprocess_progress.pack_forget()
        self.preprocess_status.pack_forget()
        self.apply_btn.configure(state="normal", text="Apply Preprocessing")
        messagebox.showerror("Preprocessing Error", f"Error during preprocessing: {error_msg}")