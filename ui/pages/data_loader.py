import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os
import numpy as np
import threading

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
        
        # Icon placeholder
        icon_label = ctk.CTkLabel(
            self.inner_frame,
            text="📁",
            font=("Segoe UI", 56),
        )
        icon_label.pack(pady=(0, 20))
        
        self.title = ctk.CTkLabel(
            self.inner_frame, 
            text="Load Dataset", 
            font=("Segoe UI", 22, "bold"),
            text_color="#111827"
        )
        self.title.pack(pady=(0, 10))
        
        self.desc = ctk.CTkLabel(
            self.inner_frame,
            text="Drag & Drop your file here or click to browse\nSupported formats: CSV, JSON, XLSX",
            font=("Segoe UI", 14),
            text_color="#6B7280",
            justify="center"
        )
        self.desc.pack(pady=(0, 25))
        
        self.browse_btn = ctk.CTkButton(
            self.inner_frame,
            text="Browse Files",
            command=self.browse_file,
            font=("Segoe UI", 14, "bold"),
            fg_color="#2563EB",
            hover_color="#1D4ED8",
            height=48,
            width=180,
            corner_radius=8
        )
        self.browse_btn.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Data Files", "*.csv *.xlsx *.json"), ("All Files", "*.*")]
        )
        if file_path:
            self.load_callback(file_path)

class TableFrame(ctk.CTkFrame):
    def __init__(self, master, df=None):
        super().__init__(master, fg_color="white", corner_radius=12, border_width=1, border_color="#E5E7EB")
        self.df = df
        self._widgets = []  # Cache for reusable widgets
        
        # Create canvas and scrollbars
        self.canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self.v_scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.h_scrollbar = ctk.CTkScrollbar(self, orientation="horizontal", command=self.canvas.xview)
        
        # Create scrollable frame inside canvas
        self.scrollable_frame = ctk.CTkFrame(self.canvas, fg_color="white")
        
        # Configure canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        # Grid layout
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Mouse wheel binding - bind only to this canvas, not all
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))
        
        if df is not None:
            self.populate_table(df)
        else:
            self.populate_initial()

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def populate_initial(self):
        columns = ["CUSTOMER ID", "AGE", "GENDER", "PLAN", "TENURE"]
        data = [
            ["CUST-001", "34", "Male", "Premium", "24 months"],
            ["CUST-002", "28", "Female", "Basic", "12 months"],
            ["CUST-003", "45", "Female", "", "60 months"],
            ["CUST-004", "51", "Male", "Premium", "120 months"],
        ]
        self.create_table_grid(columns, data)

    def populate_table(self, df):
        # Limit rows and columns for performance
        limit_rows = 100
        limit_cols = 15
        df_head = df.head(limit_rows)
        if df_head.shape[1] > limit_cols:
            df_head = df_head.iloc[:, :limit_cols]
        columns = [str(c).upper()[:25] for c in df_head.columns]  # Truncate column names
        data = df_head.astype(str).values.tolist()
        self.create_table_grid(columns, data)

    def create_table_grid(self, columns, data):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        # Headers
        for i, col in enumerate(columns):
            label = ctk.CTkLabel(
                self.scrollable_frame, 
                text=col, 
                font=("Segoe UI", 11, "bold"), 
                text_color="#374151",
                fg_color="#F9FAFB",
                padx=12,
                pady=10,
                anchor="w",
                width=120
            )
            label.grid(row=0, column=i, sticky="ew", padx=(0, 1), pady=(0, 1))
            
        # Data - batch create for performance
        for r, row in enumerate(data):
            for c, val in enumerate(row):
                if val == "nan": val = ""
                val = val[:30] + "..." if len(val) > 30 else val  # Truncate long values
                bg = "white" if r % 2 == 0 else "#FAFAFA"
                label = ctk.CTkLabel(
                    self.scrollable_frame,
                    text=val,
                    font=("Segoe UI", 11),
                    text_color="#4B5563",
                    fg_color=bg,
                    padx=12,
                    pady=8,
                    anchor="w",
                    width=120
                )
                label.grid(row=r+1, column=c, sticky="ew", padx=(0, 1), pady=(0, 1))

class DataLoaderPage(ctk.CTkFrame):
    def __init__(self, master, app_instance):
        super().__init__(master, fg_color="#F9FAFB", corner_radius=0)
        self.app = app_instance
        self.preprocessing_options = {}
        self.setup_ui()
        
        # Restore state if data already loaded
        self.after(100, self._restore_state)
    
    def _restore_state(self):
        """Restore UI state if data was previously loaded"""
        df = self.app.get_dataframe()
        if df is not None and self.app.file_path:
            self.update_ui(df, self.app.file_path)
        
    def setup_ui(self):
        # Header
        header = ctk.CTkLabel(
            self,
            text="Data Loading & Preprocessing",
            font=("Segoe UI", 26, "bold"),
            text_color="#0F172A",
            anchor="w"
        )
        header.pack(padx=30, pady=(30, 5), anchor="w")
        
        subtitle = ctk.CTkLabel(
            self,
            text="Load your dataset, preview data, and apply preprocessing steps before analysis.",
            font=("Segoe UI", 14),
            text_color="#64748B",
            anchor="w"
        )
        subtitle.pack(padx=30, pady=(0, 20), anchor="w")

        # View Switcher
        self.view_var = ctk.StringVar(value="Load Data")
        self.view_switcher = ctk.CTkSegmentedButton(
            self, 
            values=["Load Data", "Preview", "Preprocessing"],
            variable=self.view_var,
            command=self.switch_view,
            font=("Segoe UI", 12, "bold"),
            height=32
        )
        self.view_switcher.pack(padx=30, pady=(0, 20), anchor="w")
        
        # Content Area
        self.content_area = ctk.CTkFrame(self, fg_color="transparent")
        self.content_area.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        
        # --- Load Data View ---
        self.load_frame = ctk.CTkFrame(
            self.content_area, 
            fg_color="white", 
            corner_radius=8, 
            border_width=1, 
            border_color="#E2E8F0"
        )
        self.create_load_view()
        
        # --- Preview View ---
        self.preview_frame = ctk.CTkFrame(
            self.content_area, 
            fg_color="white", 
            corner_radius=8, 
            border_width=1, 
            border_color="#E2E8F0"
        )
        self.create_preview_view()
        
        # --- Preprocessing View ---
        self.preprocess_frame = ctk.CTkFrame(
            self.content_area, 
            fg_color="white", 
            corner_radius=8, 
            border_width=1, 
            border_color="#E2E8F0"
        )
        self.create_preprocessing_view()
        
        # Initialize view
        self.switch_view("Load Data")

    def switch_view(self, view_name):
        self.load_frame.pack_forget()
        self.preview_frame.pack_forget()
        self.preprocess_frame.pack_forget()
        
        if view_name == "Load Data":
            self.load_frame.pack(fill="both", expand=True)
        elif view_name == "Preview":
            self.preview_frame.pack(fill="both", expand=True)
            # Refresh table when switching to preview
            df = self.app.get_dataframe()
            if df is not None:
                self.table_frame.populate_table(df)
        elif view_name == "Preprocessing":
            self.preprocess_frame.pack(fill="both", expand=True)

    def create_load_view(self):
        """Create the Load Data view"""
        inner = ctk.CTkFrame(self.load_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=40, pady=40)
        
        # File Input Container
        self.file_input_container = ctk.CTkFrame(inner, fg_color="transparent")
        self.file_input_container.pack(fill="x", pady=(0, 30))
        
        # Drop Zone
        self.drop_zone = DropZone(self.file_input_container, self.load_file)
        self.drop_zone.pack(fill="x")
        
        # File Widget (shown after file is loaded)
        self.file_widget = ctk.CTkFrame(self.file_input_container, fg_color="#F0FDF4", corner_radius=10, border_width=1, border_color="#BBF7D0")
        
        file_inner = ctk.CTkFrame(self.file_widget, fg_color="transparent")
        file_inner.pack(fill="x", padx=20, pady=16)
        
        self.file_icon = ctk.CTkLabel(
            file_inner,
            text="✓",
            text_color="#16A34A",
            font=("Segoe UI", 20, "bold")
        )
        self.file_icon.pack(side="left", padx=(0, 12))
        
        file_info = ctk.CTkFrame(file_inner, fg_color="transparent")
        file_info.pack(side="left", fill="x", expand=True)
        
        self.file_label = ctk.CTkLabel(
            file_info,
            text="...",
            text_color="#111827",
            font=("Segoe UI", 14, "bold"),
            anchor="w"
        )
        self.file_label.pack(anchor="w")
        
        self.file_size_label = ctk.CTkLabel(
            file_info,
            text="",
            text_color="#6B7280",
            font=("Segoe UI", 11),
            anchor="w"
        )
        self.file_size_label.pack(anchor="w")
        
        close_btn = ctk.CTkButton(
            file_inner,
            text="×",
            width=32,
            height=32,
            fg_color="#FEE2E2",
            text_color="#DC2626",
            hover_color="#FECACA",
            font=("Segoe UI", 20),
            corner_radius=6,
            command=self.clear_file
        )
        close_btn.pack(side="right")
        
        # Stats Cards
        stats_label = ctk.CTkLabel(
            inner,
            text="Dataset Overview",
            font=("Segoe UI", 18, "bold"),
            text_color="#111827",
            anchor="w"
        )
        stats_label.pack(fill="x", pady=(0, 16))
        
        stats_frame = ctk.CTkFrame(inner, fg_color="transparent")
        stats_frame.pack(fill="x", pady=(0, 20))
        
        self.rows_card = self.create_stat_card(stats_frame, "📊 Rows", "0")
        self.rows_card.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.cols_card = self.create_stat_card(stats_frame, "📋 Columns", "0")
        self.cols_card.pack(side="left", fill="x", expand=True, padx=(10, 10))
        
        self.missing_card = self.create_stat_card(stats_frame, "⚠️ Missing", "0")
        self.missing_card.pack(side="left", fill="x", expand=True, padx=(10, 10))
        
        self.duplicates_card = self.create_stat_card(stats_frame, "🔄 Duplicates", "0")
        self.duplicates_card.pack(side="left", fill="x", expand=True, padx=(10, 0))
        
        # Status Message
        self.status_frame = ctk.CTkFrame(inner, fg_color="#FEF3C7", corner_radius=8)
        self.status_frame.pack(fill="x", pady=(20, 0))
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="📁 No data loaded yet. Click 'Browse Files' to get started.",
            fg_color="transparent",
            text_color="#92400E",
            font=("Segoe UI", 13),
            anchor="w"
        )
        self.status_label.pack(fill="x", padx=16, pady=14)

    def create_preview_view(self):
        """Create the Preview view with data table"""
        inner = ctk.CTkFrame(self.preview_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header with row count
        header_frame = ctk.CTkFrame(inner, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 16))
        
        ctk.CTkLabel(
            header_frame,
            text="Data Preview",
            font=("Segoe UI", 18, "bold"),
            text_color="#111827",
            anchor="w"
        ).pack(side="left")
        
        self.preview_info = ctk.CTkLabel(
            header_frame,
            text="Showing first 100 rows",
            font=("Segoe UI", 12),
            text_color="#6B7280",
            anchor="e"
        )
        self.preview_info.pack(side="right")
        
        # Table
        self.table_frame = TableFrame(inner)
        self.table_frame.pack(fill="both", expand=True)

    def create_preprocessing_view(self):
        """Create the Preprocessing view"""
        inner = ctk.CTkFrame(self.preprocess_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=40, pady=40)
        
        # Header
        ctk.CTkLabel(
            inner,
            text="Preprocessing Options",
            font=("Segoe UI", 18, "bold"),
            text_color="#111827",
            anchor="w"
        ).pack(fill="x", pady=(0, 8))
        
        ctk.CTkLabel(
            inner,
            text="Select preprocessing steps to apply to your dataset",
            font=("Segoe UI", 13),
            text_color="#6B7280",
            anchor="w"
        ).pack(fill="x", pady=(0, 24))
        
        # Preprocessing options in a 2-column grid
        options_frame = ctk.CTkFrame(inner, fg_color="transparent")
        options_frame.pack(fill="x", pady=(0, 30))
        options_frame.grid_columnconfigure(0, weight=1)
        options_frame.grid_columnconfigure(1, weight=1)
        
        # Row 1
        self.create_preprocess_option(options_frame, "Missing Values", 
            ["None", "Remove Rows", "Mean", "Median", "Mode", "Forward", "Backward"], 0, 0)
        
        self.create_preprocess_option(options_frame, "Duplicates", 
            ["None", "Remove All", "Keep First", "Keep Last"], 0, 1)
        
        # Row 2
        self.create_preprocess_option(options_frame, "Normalization", 
            ["None", "Min-Max", "Z-Score", "Robust"], 1, 0)
        
        self.create_preprocess_option(options_frame, "Outliers", 
            ["None", "Remove IQR", "Cap IQR", "Z-Score >3"], 1, 1)
        
        # Row 3
        self.create_preprocess_option(options_frame, "Encoding", 
            ["None", "One-Hot", "Label"], 2, 0)
        
        # Progress bar
        self.preprocess_progress = ctk.CTkProgressBar(inner, mode="indeterminate", height=4)
        
        self.preprocess_status = ctk.CTkLabel(
            inner,
            text="",
            text_color="#6B7280",
            font=("Segoe UI", 12)
        )
        
        # Apply Button
        btn_frame = ctk.CTkFrame(inner, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(0, 20))
        
        self.apply_btn = ctk.CTkButton(
            btn_frame,
            text="Apply Preprocessing",
            command=self.apply_preprocessing,
            font=("Segoe UI", 14, "bold"),
            fg_color="#059669",
            hover_color="#047857",
            height=48,
            width=220,
            corner_radius=8
        )
        self.apply_btn.pack(side="left")
        
        # Applied Steps Section
        steps_frame = ctk.CTkFrame(inner, fg_color="transparent")
        steps_frame.pack(fill="both", expand=True)
        
        ctk.CTkLabel(
            steps_frame,
            text="Applied Steps",
            font=("Segoe UI", 16, "bold"),
            text_color="#111827",
            anchor="w"
        ).pack(fill="x", pady=(0, 12))
        
        self.steps_container = ctk.CTkFrame(steps_frame, fg_color="#F9FAFB", corner_radius=8)
        self.steps_container.pack(fill="both", expand=True)
        
        self.steps_label = ctk.CTkLabel(
            self.steps_container,
            text="No preprocessing steps applied yet.",
            text_color="#6B7280",
            font=("Segoe UI", 13),
            anchor="w"
        )
        self.steps_label.pack(fill="x", padx=16, pady=16)

    def create_preprocess_option(self, parent, title, options, row, col):
        """Create a preprocessing option card"""
        card = ctk.CTkFrame(parent, fg_color="#F9FAFB", corner_radius=8)
        card.grid(row=row, column=col, padx=8, pady=8, sticky="ew")
        
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="x", padx=16, pady=14)
        
        # Title
        ctk.CTkLabel(
            inner,
            text=title,
            font=("Segoe UI", 13, "bold"),
            text_color="#374151",
            anchor="w"
        ).pack(fill="x", pady=(0, 8))
        
        # Dropdown
        var = ctk.StringVar(value=options[0])
        dropdown = ctk.CTkOptionMenu(
            inner,
            values=options,
            variable=var,
            fg_color="#2563EB",
            button_color="#2563EB",
            button_hover_color="#1D4ED8",
            dropdown_fg_color="white",
            dropdown_hover_color="#F3F4F6",
            dropdown_text_color="#111827",
            font=("Segoe UI", 12),
            height=36,
            corner_radius=6
        )
        dropdown.pack(fill="x")
        
        self.preprocessing_options[title] = var

    def create_stat_card(self, parent, title, value):
        card = ctk.CTkFrame(parent, fg_color="#F9FAFB", corner_radius=10, border_width=1, border_color="#E5E7EB")
        
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=14)
        
        ctk.CTkLabel(
            inner,
            text=title,
            text_color="#6B7280",
            font=("Segoe UI", 12),
            anchor="w"
        ).pack(anchor="w", pady=(0, 4))
        
        val_label = ctk.CTkLabel(
            inner,
            text=value,
            text_color="#111827",
            font=("Segoe UI", 24, "bold"),
            anchor="w"
        )
        val_label.pack(anchor="w")
        
        card.value_label = val_label
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
        self.update_ui(df, file_path)
    
    def _handle_load_error(self, error_msg):
        self.drop_zone.configure(fg_color="#FAFAFA", border_color="#D1D5DB")
        self.drop_zone.browse_btn.configure(state="normal", text="Browse Files")
        messagebox.showerror("Load Error", f"Error loading file: {error_msg}")

    def update_ui(self, df, file_path):
        if df is None: return
        
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
        
        self.file_label.configure(text=filename)
        self.file_size_label.configure(text=f"{size_str} • {df.shape[0]} rows × {df.shape[1]} columns")
        
        self.drop_zone.pack_forget()
        self.file_widget.pack(fill="x")
        
        rows, cols = df.shape
        missing = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()
        
        self.rows_card.value_label.configure(text=f"{rows:,}")
        self.cols_card.value_label.configure(text=str(cols))
        self.missing_card.value_label.configure(text=str(missing))
        self.duplicates_card.value_label.configure(text=str(duplicates))
        
        self.table_frame.populate_table(df)
        self.preview_info.configure(text=f"Showing first {min(100, rows)} rows of {rows:,}")
        
        self.status_frame.configure(fg_color="#ECFDF5")
        self.status_label.configure(
            text=f"✓ Loaded '{filename}' successfully. Switch to 'Preview' tab to see the data.",
            text_color="#059669"
        )

    def clear_file(self):
        self.app.set_dataframe(None, None)
        self.file_widget.pack_forget()
        self.drop_zone.pack(fill="x")
        
        self.rows_card.value_label.configure(text="0")
        self.cols_card.value_label.configure(text="0")
        self.missing_card.value_label.configure(text="0")
        self.duplicates_card.value_label.configure(text="0")
        
        self.table_frame.populate_initial()
        self.preview_info.configure(text="Showing first 100 rows")
        
        self.status_frame.configure(fg_color="#FEF3C7")
        self.status_label.configure(
            text="📁 No data loaded yet. Click 'Browse Files' to get started.",
            text_color="#92400E"
        )
        
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
        
        options = {key: var.get() for key, var in self.preprocessing_options.items()}
        
        thread = threading.Thread(target=self._apply_preprocessing_thread, args=(df.copy(), options))
        thread.daemon = True
        thread.start()
    
    def _apply_preprocessing_thread(self, df, options):
        df_processed = df.copy()
        steps_applied = []
        
        try:
            # Missing Values
            missing_method = self.preprocessing_options["Missing Values"].get()
            if missing_method != "None":
                if missing_method == "Remove Rows":
                    df_processed = df_processed.dropna()
                    steps_applied.append(f"✓ Removed rows with missing values")
                elif missing_method in ["Fill with Mean", "Mean"]:
                    df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
                    steps_applied.append(f"✓ Filled missing values with mean")
                elif missing_method in ["Fill with Median", "Median"]:
                    df_processed = df_processed.fillna(df_processed.median(numeric_only=True))
                    steps_applied.append(f"✓ Filled missing values with median")
                elif missing_method in ["Fill with Mode", "Mode"]:
                    df_processed = df_processed.fillna(df_processed.mode().iloc[0])
                    steps_applied.append(f"✓ Filled missing values with mode")
                elif missing_method in ["Forward Fill", "Forward"]:
                    df_processed = df_processed.ffill()
                    steps_applied.append(f"✓ Forward filled missing values")
                elif missing_method in ["Backward Fill", "Backward"]:
                    df_processed = df_processed.bfill()
                    steps_applied.append(f"✓ Backward filled missing values")
            
            # Duplicates
            dup_method = self.preprocessing_options["Duplicates"].get()
            if dup_method != "None":
                if dup_method == "Remove All":
                    df_processed = df_processed.drop_duplicates()
                    steps_applied.append(f"✓ Removed all duplicate rows")
                elif dup_method == "Keep First":
                    df_processed = df_processed.drop_duplicates(keep='first')
                    steps_applied.append(f"✓ Removed duplicates, kept first")
                elif dup_method == "Keep Last":
                    df_processed = df_processed.drop_duplicates(keep='last')
                    steps_applied.append(f"✓ Removed duplicates, kept last")
            
            # Normalization
            norm_method = self.preprocessing_options["Normalization"].get()
            if norm_method != "None":
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    if norm_method in ["Min-Max (0-1)", "Min-Max"]:
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
                        steps_applied.append(f"✓ Applied Min-Max normalization")
                    elif norm_method == "Z-Score":
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
                        steps_applied.append(f"✓ Applied Z-Score normalization")
                    elif norm_method in ["Robust Scaler", "Robust"]:
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
                        steps_applied.append(f"✓ Applied Robust scaling")
            
            # Outlier Handling
            outlier_method = self.preprocessing_options.get("Outliers", ctk.StringVar(value="None")).get()
            if outlier_method != "None":
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    if outlier_method == "Remove IQR":
                        for col in numeric_cols:
                            Q1 = df_processed[col].quantile(0.25)
                            Q3 = df_processed[col].quantile(0.75)
                            IQR = Q3 - Q1
                            df_processed = df_processed[(df_processed[col] >= Q1 - 1.5*IQR) & 
                                                       (df_processed[col] <= Q3 + 1.5*IQR)]
                        steps_applied.append(f"✓ Removed outliers using IQR")
                    elif outlier_method == "Cap IQR":
                        for col in numeric_cols:
                            Q1 = df_processed[col].quantile(0.25)
                            Q3 = df_processed[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - 1.5*IQR
                            upper = Q3 + 1.5*IQR
                            df_processed[col] = df_processed[col].clip(lower, upper)
                        steps_applied.append(f"✓ Capped outliers using IQR")
                    elif outlier_method in ["Remove Z-Score (>3)", "Z-Score >3"]:
                        from scipy import stats
                        for col in numeric_cols:
                            z_scores = np.abs(stats.zscore(df_processed[col].dropna()))
                            df_processed = df_processed[(z_scores < 3)]
                        steps_applied.append(f"✓ Removed outliers using Z-Score")
            
            # Encoding
            encoding_method = self.preprocessing_options["Encoding"].get()
            if encoding_method != "None":
                cat_cols = df_processed.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    if encoding_method == "One-Hot":
                        df_processed = pd.get_dummies(df_processed, columns=cat_cols)
                        steps_applied.append(f"✓ Applied One-Hot encoding")
                    elif encoding_method in ["Label Encoding", "Label"]:
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        for col in cat_cols:
                            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                        steps_applied.append(f"✓ Applied Label encoding")
            
            self.after(0, lambda: self._finish_preprocessing(df_processed, steps_applied))
        except Exception as e:
            self.after(0, lambda err=str(e): self._handle_preprocessing_error(err))
    
    def _finish_preprocessing(self, df_processed, steps_applied):
        self.preprocess_progress.stop()
        self.preprocess_progress.pack_forget()
        self.preprocess_status.configure(text="✓ Preprocessing complete!")
        self.after(2000, lambda: self.preprocess_status.pack_forget())
        self.apply_btn.configure(state="normal", text="Apply Preprocessing")
        
        self.app.set_dataframe(df_processed, self.app.file_path)
        self.update_ui(df_processed, self.app.file_path)
        
        if steps_applied:
            steps_text = "\n".join(steps_applied)
            self.steps_label.configure(text=steps_text, text_color="#059669")
            messagebox.showinfo("Success", f"Applied {len(steps_applied)} preprocessing step(s)!")
        else:
            messagebox.showinfo("No Changes", "No preprocessing options were selected.")
    
    def _handle_preprocessing_error(self, error_msg):
        self.preprocess_progress.stop()
        self.preprocess_progress.pack_forget()
        self.preprocess_status.pack_forget()
        self.apply_btn.configure(state="normal", text="Apply Preprocessing")
        messagebox.showerror("Preprocessing Error", f"Error during preprocessing: {error_msg}")