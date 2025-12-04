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
        self.configure(height=240)
        
        self.setup_ui()
        
    def setup_ui(self):
        self.inner_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.inner_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Icon placeholder
        icon_label = ctk.CTkLabel(
            self.inner_frame,
            text="📁",
            font=("Segoe UI", 48),
        )
        icon_label.pack(pady=(0, 15))
        
        self.title = ctk.CTkLabel(
            self.inner_frame, 
            text="Load Dataset", 
            font=("Segoe UI", 20, "bold"),
            text_color="#111827"
        )
        self.title.pack(pady=(0, 8))
        
        self.desc = ctk.CTkLabel(
            self.inner_frame,
            text="Drag & Drop your file here or click to browse\nSupported formats: CSV, JSON, XLSX",
            font=("Segoe UI", 13),
            text_color="#6B7280",
            justify="center"
        )
        self.desc.pack(pady=(0, 20))
        
        self.browse_btn = ctk.CTkButton(
            self.inner_frame,
            text="Browse Files",
            command=self.browse_file,
            font=("Segoe UI", 14, "bold"),
            fg_color="#2563EB",
            hover_color="#1D4ED8",
            height=44,
            width=160,
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
        
        # Mouse wheel binding
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
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
        limit = 100
        df_head = df.head(limit)
        columns = [str(c).upper() for c in df.columns]
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
                font=("Segoe UI", 12, "bold"), 
                text_color="#374151",
                fg_color="#F9FAFB",
                padx=16,
                pady=12,
                anchor="w",
                width=140
            )
            label.grid(row=0, column=i, sticky="ew", padx=(0, 1), pady=(0, 1))
            
        # Data
        for r, row in enumerate(data):
            for c, val in enumerate(row):
                if val == "nan": val = ""
                bg = "white" if r % 2 == 0 else "#FAFAFA"
                label = ctk.CTkLabel(
                    self.scrollable_frame,
                    text=val,
                    font=("Segoe UI", 12),
                    text_color="#4B5563",
                    fg_color=bg,
                    padx=16,
                    pady=10,
                    anchor="w",
                    width=140
                )
                label.grid(row=r+1, column=c, sticky="ew", padx=(0, 1), pady=(0, 1))

class DataLoaderPage(ctk.CTkFrame):
    def __init__(self, master, app_instance):
        super().__init__(master, fg_color="#F9FAFB", corner_radius=0)
        self.app = app_instance
        self.preprocessing_options = {}
        self.setup_ui()
        
    def setup_ui(self):
        # Header with better spacing
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(padx=40, pady=(40, 0), fill="x")
        
        header = ctk.CTkLabel(
            header_frame,
            text="Data Loading & Preprocessing",
            font=("Segoe UI", 28, "bold"),
            text_color="#111827",
            anchor="w"
        )
        header.pack(anchor="w")
        
        subtitle = ctk.CTkLabel(
            header_frame,
            text="Load your dataset and apply preprocessing steps before analysis",
            font=("Segoe UI", 14),
            text_color="#6B7280",
            anchor="w"
        )
        subtitle.pack(anchor="w", pady=(6, 0))
        
        # Divider
        divider = ctk.CTkFrame(self, fg_color="#E5E7EB", height=1)
        divider.pack(fill="x", padx=40, pady=(25, 30))
        
        # Main Layout with better proportions
        main_layout = ctk.CTkFrame(self, fg_color="transparent")
        main_layout.pack(padx=40, pady=0, fill="both", expand=True)
        
        # Left Panel (40% width)
        left_panel = self.create_left_panel(main_layout)
        left_panel.pack(side="left", fill="both", expand=False, padx=(0, 15), ipadx=280)
        
        # Right Panel (60% width)
        right_panel = self.create_right_panel(main_layout)
        right_panel.pack(side="right", fill="both", expand=True, padx=(15, 0))

    def create_left_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        
        # File Input Container
        self.file_input_container = ctk.CTkFrame(panel, fg_color="transparent")
        self.file_input_container.pack(fill="x", pady=(0, 25))
        
        # Drop Zone
        self.drop_zone = DropZone(self.file_input_container, self.load_file)
        self.drop_zone.pack(fill="x")
        
        # File Widget
        self.file_widget = ctk.CTkFrame(self.file_input_container, fg_color="white", corner_radius=10, border_width=1, border_color="#E5E7EB")
        
        file_inner = ctk.CTkFrame(self.file_widget, fg_color="transparent")
        file_inner.pack(fill="x", padx=16, pady=14)
        
        self.file_label = ctk.CTkLabel(
            file_inner,
            text="📄  ...",
            text_color="#111827",
            font=("Segoe UI", 13, "bold")
        )
        self.file_label.pack(side="left")
        
        close_btn = ctk.CTkButton(
            file_inner,
            text="×",
            width=28,
            height=28,
            fg_color="#F3F4F6",
            text_color="#6B7280",
            hover_color="#E5E7EB",
            font=("Segoe UI", 18),
            corner_radius=6,
            command=self.clear_file
        )
        close_btn.pack(side="right")
        
        # Section Header
        section_header = ctk.CTkLabel(
            panel,
            text="Preprocessing Options",
            font=("Segoe UI", 16, "bold"),
            text_color="#111827",
            anchor="w"
        )
        section_header.pack(fill="x", pady=(0, 16))
        
        # Create preprocessing sections with better spacing
        self.create_preprocess_section(panel, "Missing Values", 
            ["None", "Remove Rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Forward Fill", "Backward Fill"])
        
        self.create_preprocess_section(panel, "Duplicates", 
            ["None", "Remove All", "Keep First", "Keep Last"])
        
        self.create_preprocess_section(panel, "Normalization", 
            ["None", "Min-Max (0-1)", "Z-Score", "Robust Scaler"])
        
        self.create_preprocess_section(panel, "Outlier Handling", 
            ["None", "Remove IQR", "Cap IQR", "Remove Z-Score (>3)"])
        
        self.create_preprocess_section(panel, "Encoding", 
            ["None", "One-Hot", "Label Encoding", "Target Encoding"])
        
        # Progress bar
        self.preprocess_progress = ctk.CTkProgressBar(panel, mode="indeterminate", height=6)
        self.preprocess_progress.pack(fill="x", pady=(20, 8))
        self.preprocess_progress.pack_forget()
        
        self.preprocess_status = ctk.CTkLabel(
            panel,
            text="",
            text_color="#6B7280",
            font=("Segoe UI", 12)
        )
        self.preprocess_status.pack(fill="x", pady=(0, 12))
        self.preprocess_status.pack_forget()
        
        # Apply Button with better prominence
        self.apply_btn = ctk.CTkButton(
            panel,
            text="Apply Preprocessing",
            command=self.apply_preprocessing,
            font=("Segoe UI", 14, "bold"),
            fg_color="#059669",
            hover_color="#047857",
            height=46,
            corner_radius=8
        )
        self.apply_btn.pack(fill="x", pady=(20, 0))
        
        return panel

    def create_preprocess_section(self, parent, title, options):
        section = ctk.CTkFrame(parent, fg_color="white", corner_radius=10, border_width=1, border_color="#E5E7EB")
        section.pack(fill="x", pady=(0, 12))
        
        # Title with better padding
        title_label = ctk.CTkLabel(
            section,
            text=title,
            font=("Segoe UI", 13, "bold"),
            text_color="#374151",
            anchor="w"
        )
        title_label.pack(fill="x", padx=16, pady=(14, 10))
        
        # Dropdown
        var = ctk.StringVar(value=options[0])
        dropdown = ctk.CTkOptionMenu(
            section,
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
            corner_radius=6,
            anchor="w"
        )
        dropdown.pack(fill="x", padx=16, pady=(0, 14))
        
        self.preprocessing_options[title] = var

    def create_right_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        
        # Summary Header
        summary_label = ctk.CTkLabel(
            panel,
            text="Dataset Overview",
            font=("Segoe UI", 16, "bold"),
            text_color="#111827",
            anchor="w"
        )
        summary_label.pack(fill="x", pady=(0, 16))
        
        # Stats with improved cards
        stats_frame = ctk.CTkFrame(panel, fg_color="transparent")
        stats_frame.pack(fill="x", pady=(0, 30))
        
        self.rows_card = self.create_stat_card(stats_frame, "Rows & Columns", "0, 0")
        self.rows_card.pack(side="left", fill="x", expand=True, padx=(0, 8))
        
        self.missing_card = self.create_stat_card(stats_frame, "Missing Values", "0")
        self.missing_card.pack(side="left", fill="x", expand=True, padx=(8, 8))
        
        self.duplicates_card = self.create_stat_card(stats_frame, "Duplicate Rows", "0")
        self.duplicates_card.pack(side="left", fill="x", expand=True, padx=(8, 0))
        
        # Preview Header
        preview_label = ctk.CTkLabel(
            panel,
            text="Data Preview",
            font=("Segoe UI", 16, "bold"),
            text_color="#111827",
            anchor="w"
        )
        preview_label.pack(fill="x", pady=(0, 16))
        
        # Table with better height
        table_container = ctk.CTkFrame(panel, fg_color="transparent", height=320)
        table_container.pack(fill="both", expand=True, pady=(0, 30))
        table_container.pack_propagate(False)
        
        self.table_frame = TableFrame(table_container)
        self.table_frame.pack(fill="both", expand=True)
        
        # Applied Steps
        steps_header = ctk.CTkLabel(
            panel,
            text="Applied Steps",
            font=("Segoe UI", 16, "bold"),
            text_color="#111827",
            anchor="w"
        )
        steps_header.pack(fill="x", pady=(0, 16))
        
        self.steps_label = ctk.CTkLabel(
            panel,
            text="No data loaded yet.",
            fg_color="#FEF3C7",
            text_color="#92400E",
            corner_radius=8,
            padx=18,
            pady=14,
            anchor="w",
            font=("Segoe UI", 12)
        )
        self.steps_label.pack(fill="x")
        
        return panel

    def create_stat_card(self, parent, title, value):
        card = ctk.CTkFrame(parent, fg_color="white", corner_radius=10, border_width=1, border_color="#E5E7EB")
        
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=18, pady=16)
        
        ctk.CTkLabel(
            inner,
            text=title,
            text_color="#6B7280",
            font=("Segoe UI", 12),
            anchor="w"
        ).pack(anchor="w", pady=(0, 6))
        
        val_label = ctk.CTkLabel(
            inner,
            text=value,
            text_color="#111827",
            font=("Segoe UI", 26, "bold"),
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
        self.file_label.configure(text=f"📄  {filename}")
        
        self.drop_zone.pack_forget()
        self.file_widget.pack(fill="x")
        
        rows, cols = df.shape
        missing = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()
        
        self.rows_card.value_label.configure(text=f"{rows}, {cols}")
        self.missing_card.value_label.configure(text=str(missing))
        self.duplicates_card.value_label.configure(text=str(duplicates))
        
        self.table_frame.populate_table(df)
        
        self.steps_label.configure(
            text=f"✓  Loaded '{filename}' successfully.",
            fg_color="#ECFDF5",
            text_color="#059669"
        )

    def clear_file(self):
        self.app.set_dataframe(None, None)
        self.file_widget.pack_forget()
        self.drop_zone.pack(fill="x")
        
        self.rows_card.value_label.configure(text="0, 0")
        self.missing_card.value_label.configure(text="0")
        self.duplicates_card.value_label.configure(text="0")
        
        self.table_frame.populate_initial()
        
        self.steps_label.configure(
            text="No data loaded yet.",
            fg_color="#FEF3C7",
            text_color="#92400E"
        )

    def apply_preprocessing(self):
        df = self.app.get_dataframe()
        if df is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        self.preprocess_progress.pack(fill="x", pady=(20, 8))
        self.preprocess_progress.start()
        self.preprocess_status.configure(text="Applying preprocessing...")
        self.preprocess_status.pack(fill="x", pady=(0, 12))
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
                    steps_applied.append(f"Removed rows with missing values")
                elif missing_method == "Fill with Mean":
                    df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
                    steps_applied.append(f"Filled missing values with mean")
                elif missing_method == "Fill with Median":
                    df_processed = df_processed.fillna(df_processed.median(numeric_only=True))
                    steps_applied.append(f"Filled missing values with median")
                elif missing_method == "Fill with Mode":
                    df_processed = df_processed.fillna(df_processed.mode().iloc[0])
                    steps_applied.append(f"Filled missing values with mode")
                elif missing_method == "Forward Fill":
                    df_processed = df_processed.fillna(method='ffill')
                    steps_applied.append(f"Forward filled missing values")
                elif missing_method == "Backward Fill":
                    df_processed = df_processed.fillna(method='bfill')
                    steps_applied.append(f"Backward filled missing values")
            
            # Duplicates
            dup_method = self.preprocessing_options["Duplicates"].get()
            if dup_method != "None":
                if dup_method == "Remove All":
                    df_processed = df_processed.drop_duplicates()
                    steps_applied.append(f"Removed all duplicate rows")
                elif dup_method == "Keep First":
                    df_processed = df_processed.drop_duplicates(keep='first')
                    steps_applied.append(f"Removed duplicates, kept first occurrence")
                elif dup_method == "Keep Last":
                    df_processed = df_processed.drop_duplicates(keep='last')
                    steps_applied.append(f"Removed duplicates, kept last occurrence")
            
            # Normalization
            norm_method = self.preprocessing_options["Normalization"].get()
            if norm_method != "None":
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    if norm_method == "Min-Max (0-1)":
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
                        steps_applied.append(f"Applied Min-Max normalization")
                    elif norm_method == "Z-Score":
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
                        steps_applied.append(f"Applied Z-Score normalization")
                    elif norm_method == "Robust Scaler":
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
                        steps_applied.append(f"Applied Robust scaling")
            
            # Outlier Handling
            outlier_method = self.preprocessing_options["Outlier Handling"].get()
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
                        steps_applied.append(f"Removed outliers using IQR method")
                    elif outlier_method == "Cap IQR":
                        for col in numeric_cols:
                            Q1 = df_processed[col].quantile(0.25)
                            Q3 = df_processed[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - 1.5*IQR
                            upper = Q3 + 1.5*IQR
                            df_processed[col] = df_processed[col].clip(lower, upper)
                        steps_applied.append(f"Capped outliers using IQR method")
                    elif outlier_method == "Remove Z-Score (>3)":
                        from scipy import stats
                        for col in numeric_cols:
                            z_scores = np.abs(stats.zscore(df_processed[col].dropna()))
                            df_processed = df_processed[(z_scores < 3)]
                        steps_applied.append(f"Removed outliers using Z-Score method")
            
            # Encoding
            encoding_method = self.preprocessing_options["Encoding"].get()
            if encoding_method != "None":
                cat_cols = df_processed.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    if encoding_method == "One-Hot":
                        df_processed = pd.get_dummies(df_processed, columns=cat_cols)
                        steps_applied.append(f"Applied One-Hot encoding")
                    elif encoding_method == "Label Encoding":
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        for col in cat_cols:
                            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                        steps_applied.append(f"Applied Label encoding")
            
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
            steps_text = "✓  " + "\n✓  ".join(steps_applied)
            self.steps_label.configure(text=steps_text)
            messagebox.showinfo("Success", f"Applied {len(steps_applied)} preprocessing step(s)!")
        else:
            messagebox.showinfo("No Changes", "No preprocessing options were selected.")
    
    def _handle_preprocessing_error(self, error_msg):
        self.preprocess_progress.stop()
        self.preprocess_progress.pack_forget()
        self.preprocess_status.pack_forget()
        self.apply_btn.configure(state="normal", text="Apply Preprocessing")
        messagebox.showerror("Preprocessing Error", f"Error during preprocessing: {error_msg}")