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
        self.configure(fg_color="#FAFAFA", border_width=2, border_color="#D1D5DB", corner_radius=12, height=280)
        self.pack_propagate(False)
        self.setup_ui()
        
    def setup_ui(self):
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(inner, text="📁", font=("Segoe UI", 56)).pack(pady=(0, 20))
        ctk.CTkLabel(inner, text="Load Dataset", font=("Segoe UI", 22, "bold"), text_color="#111827").pack(pady=(0, 10))
        ctk.CTkLabel(inner, text="Drag & Drop your file here or click to browse\nSupported formats: CSV, JSON, XLSX", font=("Segoe UI", 14), text_color="#6B7280", justify="center").pack(pady=(0, 25))
        self.browse_btn = ctk.CTkButton(inner, text="Browse Files", command=self.browse_file, font=("Segoe UI", 14, "bold"), fg_color="#2563EB", hover_color="#1D4ED8", height=48, width=180, corner_radius=8)
        self.browse_btn.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Data Files", "*.csv *.xlsx *.json"), ("All Files", "*.*")])
        if file_path:
            self.load_callback(file_path)


class TableFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="white", corner_radius=12, border_width=1, border_color="#E5E7EB")
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
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.populate_initial()

    def populate_initial(self):
        self._create_table(["COL1", "COL2", "COL3"], [["--", "--", "--"], ["--", "--", "--"]])

    def populate_table(self, df):
        df_head = df.head(50).iloc[:, :10]  # Limit to 50 rows, 10 cols for speed
        cols = [str(c).upper()[:20] for c in df_head.columns]
        data = df_head.astype(str).values.tolist()
        self._create_table(cols, data)

    def _create_table(self, columns, data):
        for w in self.scrollable_frame.winfo_children():
            w.destroy()
        for i, col in enumerate(columns):
            ctk.CTkLabel(self.scrollable_frame, text=col, font=("Segoe UI", 10, "bold"), text_color="#374151", fg_color="#F9FAFB", padx=8, pady=6, anchor="w", width=100).grid(row=0, column=i, sticky="ew", padx=(0,1), pady=(0,1))
        for r, row in enumerate(data):
            for c, val in enumerate(row):
                if val == "nan": val = ""
                val = val[:25] + "..." if len(val) > 25 else val
                bg = "white" if r % 2 == 0 else "#FAFAFA"
                ctk.CTkLabel(self.scrollable_frame, text=val, font=("Segoe UI", 10), text_color="#4B5563", fg_color=bg, padx=8, pady=4, anchor="w", width=100).grid(row=r+1, column=c, sticky="ew", padx=(0,1), pady=(0,1))


class DataLoaderPage(ctk.CTkFrame):
    def __init__(self, master, app_instance):
        super().__init__(master, fg_color="#F9FAFB", corner_radius=0)
        self.app = app_instance
        self.column_options = {}
        self.missing_details_expanded = True
        self.original_stats = None
        self._cached_empty = {}
        self._columns_built = False
        self.setup_ui()
        self.after(200, self._restore_state)
    
    def _restore_state(self):
        df = self.app.get_dataframe()
        if df is not None and self.app.file_path:
            self._cached_empty = self._count_empty_fast(df)
            self.update_ui(df, self.app.file_path)

    def _count_empty_fast(self, df):
        """Optimized empty value counting"""
        result = {}
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            placeholder_count = 0
            if df[col].dtype == 'object':
                placeholder_count = int(df[col].dropna().astype(str).str.strip().isin(EMPTY_VALUE_PLACEHOLDERS).sum())
            total = null_count + placeholder_count
            if total > 0:
                result[col] = total
        return result
        
    def setup_ui(self):
        # Header
        ctk.CTkLabel(self, text="Data Loading & Preprocessing", font=("Segoe UI", 24, "bold"), text_color="#0F172A", anchor="w").pack(padx=30, pady=(25, 5), anchor="w")
        ctk.CTkLabel(self, text="Load your dataset, preview data, and apply preprocessing steps.", font=("Segoe UI", 13), text_color="#64748B", anchor="w").pack(padx=30, pady=(0, 15), anchor="w")

        self.view_var = ctk.StringVar(value="Load Data")
        self.view_switcher = ctk.CTkSegmentedButton(self, values=["Load Data", "Preview", "Preprocessing"], variable=self.view_var, command=self.switch_view, font=("Segoe UI", 12, "bold"), height=32)
        self.view_switcher.pack(padx=30, pady=(0, 15), anchor="w")
        
        self.content_area = ctk.CTkFrame(self, fg_color="transparent")
        self.content_area.pack(fill="both", expand=True, padx=30, pady=(0, 20))
        
        # Create frames
        self.load_frame = ctk.CTkFrame(self.content_area, fg_color="white", corner_radius=8, border_width=1, border_color="#E2E8F0")
        self.preview_frame = ctk.CTkFrame(self.content_area, fg_color="white", corner_radius=8, border_width=1, border_color="#E2E8F0")
        self.preprocess_frame = ctk.CTkFrame(self.content_area, fg_color="white", corner_radius=8, border_width=1, border_color="#E2E8F0")
        
        self._create_load_view()
        self._create_preview_view()
        self._create_preprocessing_view()
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
                self.after_idle(lambda: self.table_frame.populate_table(df))
        elif view_name == "Preprocessing":
            self.preprocess_frame.pack(fill="both", expand=True)
            if not self._columns_built:
                self.after_idle(self._build_column_options)

    def _create_load_view(self):
        inner = ctk.CTkFrame(self.load_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=30, pady=30)
        
        # File input
        self.file_container = ctk.CTkFrame(inner, fg_color="transparent")
        self.file_container.pack(fill="x", pady=(0, 20))
        
        self.drop_zone = DropZone(self.file_container, self.load_file)
        self.drop_zone.pack(fill="x")
        
        # File loaded widget
        self.file_widget = ctk.CTkFrame(self.file_container, fg_color="#F0FDF4", corner_radius=10, border_width=1, border_color="#BBF7D0")
        fw_inner = ctk.CTkFrame(self.file_widget, fg_color="transparent")
        fw_inner.pack(fill="x", padx=16, pady=12)
        ctk.CTkLabel(fw_inner, text="✓", text_color="#16A34A", font=("Segoe UI", 18, "bold")).pack(side="left", padx=(0, 10))
        file_info = ctk.CTkFrame(fw_inner, fg_color="transparent")
        file_info.pack(side="left", fill="x", expand=True)
        self.file_label = ctk.CTkLabel(file_info, text="", text_color="#111827", font=("Segoe UI", 13, "bold"), anchor="w")
        self.file_label.pack(anchor="w")
        self.file_size_label = ctk.CTkLabel(file_info, text="", text_color="#6B7280", font=("Segoe UI", 10), anchor="w")
        self.file_size_label.pack(anchor="w")
        ctk.CTkButton(fw_inner, text="×", width=28, height=28, fg_color="#FEE2E2", text_color="#DC2626", hover_color="#FECACA", font=("Segoe UI", 16), corner_radius=6, command=self.clear_file).pack(side="right")
        
        # Stats row
        self.overview = ctk.CTkFrame(inner, fg_color="#F9FAFB", corner_radius=8, border_width=1, border_color="#E5E7EB")
        self.overview.pack(fill="x", pady=(0, 15))
        ov_inner = ctk.CTkFrame(self.overview, fg_color="transparent")
        ov_inner.pack(fill="x", padx=16, pady=12)
        
        ctk.CTkLabel(ov_inner, text="📈 Dataset Overview", font=("Segoe UI", 14, "bold"), text_color="#111827").pack(anchor="w", pady=(0, 10))
        
        stats = ctk.CTkFrame(ov_inner, fg_color="transparent")
        stats.pack(fill="x")
        self.rows_lbl = self._stat_card(stats, "Rows", "0")
        self.cols_lbl = self._stat_card(stats, "Columns", "0")
        self.empty_lbl = self._stat_card(stats, "Empty", "0")
        self.dups_lbl = self._stat_card(stats, "Duplicates", "0")
        
        # Empty values section
        self.empty_section = ctk.CTkFrame(ov_inner, fg_color="#FEF9C3", corner_radius=6)
        self.empty_section.pack(fill="x", pady=(10, 0))
        es_header = ctk.CTkFrame(self.empty_section, fg_color="transparent")
        es_header.pack(fill="x", padx=10, pady=(8, 0))
        self.empty_toggle = ctk.CTkButton(es_header, text="▼ Empty Values by Column", font=("Segoe UI", 11, "bold"), text_color="#92400E", fg_color="transparent", hover_color="#FEF08A", anchor="w", height=20, command=self._toggle_empty)
        self.empty_toggle.pack(side="left", fill="x", expand=True)
        self.empty_details = ctk.CTkFrame(self.empty_section, fg_color="#FFFBEB", corner_radius=4)
        self.empty_details.pack(fill="x", padx=10, pady=(5, 10))
        self.empty_content = ctk.CTkLabel(self.empty_details, text="📊 Load data to see empty values", font=("Segoe UI", 11), text_color="#78716C", anchor="w")
        self.empty_content.pack(fill="x", padx=10, pady=8)
        
        # Status
        self.status_frame = ctk.CTkFrame(inner, fg_color="#FEF3C7", corner_radius=6)
        self.status_frame.pack(fill="x")
        self.status_label = ctk.CTkLabel(self.status_frame, text="📁 No data loaded yet.", fg_color="transparent", text_color="#92400E", font=("Segoe UI", 12), anchor="w")
        self.status_label.pack(fill="x", padx=12, pady=10)

    def _stat_card(self, parent, title, value):
        card = ctk.CTkFrame(parent, fg_color="white", corner_radius=6, border_width=1, border_color="#E5E7EB")
        card.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkLabel(card, text=title, text_color="#6B7280", font=("Segoe UI", 10), anchor="w").pack(anchor="w", padx=10, pady=(8, 0))
        lbl = ctk.CTkLabel(card, text=value, text_color="#111827", font=("Segoe UI", 18, "bold"), anchor="w")
        lbl.pack(anchor="w", padx=10, pady=(0, 8))
        return lbl

    def _toggle_empty(self):
        if self.missing_details_expanded:
            self.empty_details.pack_forget()
            self.empty_toggle.configure(text="▶ Empty Values by Column")
            self.missing_details_expanded = False
        else:
            self.empty_details.pack(fill="x", padx=10, pady=(5, 10))
            self.empty_toggle.configure(text="▼ Empty Values by Column")
            self.missing_details_expanded = True

    def _create_preview_view(self):
        inner = ctk.CTkFrame(self.preview_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=15, pady=15)
        header = ctk.CTkFrame(inner, fg_color="transparent")
        header.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(header, text="Data Preview", font=("Segoe UI", 16, "bold"), text_color="#111827").pack(side="left")
        self.preview_info = ctk.CTkLabel(header, text="", font=("Segoe UI", 11), text_color="#6B7280")
        self.preview_info.pack(side="right")
        self.table_frame = TableFrame(inner)
        self.table_frame.pack(fill="both", expand=True)

    def _create_preprocessing_view(self):
        inner = ctk.CTkFrame(self.preprocess_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        ctk.CTkLabel(inner, text="Column-wise Preprocessing", font=("Segoe UI", 16, "bold"), text_color="#111827").pack(anchor="w", pady=(0, 5))
        ctk.CTkLabel(inner, text="Choose preprocessing for each column", font=("Segoe UI", 11), text_color="#6B7280").pack(anchor="w", pady=(0, 10))
        
        # Global options
        global_row = ctk.CTkFrame(inner, fg_color="#F1F5F9", corner_radius=6)
        global_row.pack(fill="x", pady=(0, 10))
        gr_inner = ctk.CTkFrame(global_row, fg_color="transparent")
        gr_inner.pack(fill="x", padx=12, pady=8)
        ctk.CTkLabel(gr_inner, text="Duplicates:", font=("Segoe UI", 11), text_color="#374151").pack(side="left", padx=(0, 5))
        self.dup_var = ctk.StringVar(value="None")
        ctk.CTkOptionMenu(gr_inner, values=["None", "Remove All", "Keep First", "Keep Last"], variable=self.dup_var, width=110, height=26, font=("Segoe UI", 10)).pack(side="left")
        
        # Column table scroll
        self.col_scroll = ctk.CTkScrollableFrame(inner, fg_color="white", corner_radius=6, border_width=1, border_color="#E5E7EB", height=250)
        self.col_scroll.pack(fill="both", expand=True, pady=(0, 10))
        
        # Header row
        hdr = ctk.CTkFrame(self.col_scroll, fg_color="#F3F4F6")
        hdr.pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(hdr, text="Column", font=("Segoe UI", 10, "bold"), text_color="#374151", width=130, anchor="w").pack(side="left", padx=8, pady=5)
        ctk.CTkLabel(hdr, text="Type", font=("Segoe UI", 10, "bold"), text_color="#374151", width=60, anchor="center").pack(side="left", padx=3, pady=5)
        ctk.CTkLabel(hdr, text="Empty", font=("Segoe UI", 10, "bold"), text_color="#374151", width=50, anchor="center").pack(side="left", padx=3, pady=5)
        ctk.CTkLabel(hdr, text="Missing", font=("Segoe UI", 10, "bold"), text_color="#374151", width=100, anchor="center").pack(side="left", padx=3, pady=5)
        ctk.CTkLabel(hdr, text="Scaling", font=("Segoe UI", 10, "bold"), text_color="#374151", width=90, anchor="center").pack(side="left", padx=3, pady=5)
        ctk.CTkLabel(hdr, text="Encode", font=("Segoe UI", 10, "bold"), text_color="#374151", width=80, anchor="center").pack(side="left", padx=3, pady=5)
        
        self.col_content = ctk.CTkFrame(self.col_scroll, fg_color="transparent")
        self.col_content.pack(fill="x")
        
        # Buttons
        btn_row = ctk.CTkFrame(inner, fg_color="transparent")
        btn_row.pack(fill="x", pady=(0, 5))
        self.apply_btn = ctk.CTkButton(btn_row, text="Apply", command=self.apply_preprocessing, font=("Segoe UI", 12, "bold"), fg_color="#059669", hover_color="#047857", height=38, width=150)
        self.apply_btn.pack(side="left")
        ctk.CTkButton(btn_row, text="Reset", command=self._reset_options, font=("Segoe UI", 11), fg_color="#6B7280", height=38, width=80).pack(side="left", padx=(8, 0))
        
        self.progress = ctk.CTkProgressBar(inner, mode="indeterminate", height=3)
        self.steps_lbl = ctk.CTkLabel(inner, text="", text_color="#059669", font=("Segoe UI", 10), anchor="w", justify="left")

    def _build_column_options(self):
        """Build column options table - called lazily"""
        for w in self.col_content.winfo_children():
            w.destroy()
        self.column_options.clear()
        
        df = self.app.get_dataframe()
        if df is None:
            ctk.CTkLabel(self.col_content, text="Load data first", font=("Segoe UI", 11), text_color="#6B7280").pack(pady=15)
            return
        
        for i, col in enumerate(df.columns[:50]):  # Limit to 50 columns for performance
            is_num = pd.api.types.is_numeric_dtype(df[col])
            empty = self._cached_empty.get(col, 0)
            
            bg = "#FFFFFF" if i % 2 == 0 else "#FAFAFA"
            row = ctk.CTkFrame(self.col_content, fg_color=bg, height=32)
            row.pack(fill="x")
            row.pack_propagate(False)
            
            # Name
            name = col[:15] + ".." if len(col) > 15 else col
            ctk.CTkLabel(row, text=name, font=("Segoe UI", 9), text_color="#111827", width=130, anchor="w").pack(side="left", padx=8)
            
            # Type
            t_color = "#3B82F6" if is_num else "#8B5CF6"
            ctk.CTkLabel(row, text="Num" if is_num else "Text", font=("Segoe UI", 8), text_color="white", fg_color=t_color, corner_radius=3, width=45).pack(side="left", padx=3)
            
            # Empty count
            e_color = "#DC2626" if empty > 0 else "#059669"
            ctk.CTkLabel(row, text=str(empty), font=("Segoe UI", 9, "bold"), text_color=e_color, width=50, anchor="center").pack(side="left", padx=3)
            
            # Missing dropdown
            m_opts = ["None", "Drop", "Mean", "Median", "Mode", "FFill", "BFill"] if is_num else ["None", "Drop", "Mode", "FFill", "BFill"]
            m_var = ctk.StringVar(value="None")
            ctk.CTkOptionMenu(row, values=m_opts, variable=m_var, width=90, height=24, font=("Segoe UI", 9), fg_color="#E5E7EB", button_color="#D1D5DB", text_color="#111827").pack(side="left", padx=3)
            
            # Scaling dropdown
            if is_num:
                s_var = ctk.StringVar(value="None")
                ctk.CTkOptionMenu(row, values=["None", "MinMax", "ZScore", "Robust"], variable=s_var, width=80, height=24, font=("Segoe UI", 9), fg_color="#E5E7EB", button_color="#D1D5DB", text_color="#111827").pack(side="left", padx=3)
            else:
                s_var = None
                ctk.CTkLabel(row, text="-", font=("Segoe UI", 9), text_color="#9CA3AF", width=80, anchor="center").pack(side="left", padx=3)
            
            # Encoding dropdown
            if not is_num:
                e_var = ctk.StringVar(value="None")
                ctk.CTkOptionMenu(row, values=["None", "Label", "OneHot"], variable=e_var, width=70, height=24, font=("Segoe UI", 9), fg_color="#E5E7EB", button_color="#D1D5DB", text_color="#111827").pack(side="left", padx=3)
            else:
                e_var = None
                ctk.CTkLabel(row, text="-", font=("Segoe UI", 9), text_color="#9CA3AF", width=70, anchor="center").pack(side="left", padx=3)
            
            self.column_options[col] = {'num': is_num, 'miss': m_var, 'scale': s_var, 'encode': e_var}
        
        if len(df.columns) > 50:
            ctk.CTkLabel(self.col_content, text=f"... and {len(df.columns) - 50} more columns", font=("Segoe UI", 9), text_color="#6B7280").pack(pady=5)
        
        self._columns_built = True

    def _reset_options(self):
        for opts in self.column_options.values():
            opts['miss'].set("None")
            if opts['scale']: opts['scale'].set("None")
            if opts['encode']: opts['encode'].set("None")
        self.dup_var.set("None")

    def load_file(self, file_path):
        self.drop_zone.browse_btn.configure(state="disabled", text="Loading...")
        threading.Thread(target=self._load_thread, args=(file_path,), daemon=True).start()
    
    def _load_thread(self, path):
        try:
            if path.endswith('.csv'):
                df = pd.read_csv(path)
            elif path.endswith('.xlsx'):
                df = pd.read_excel(path)
            elif path.endswith('.json'):
                df = pd.read_json(path)
            else:
                df = pd.read_csv(path)
            self._cached_empty = self._count_empty_fast(df)
            self.after(0, lambda: self._finish_load(df, path))
        except Exception as e:
            self.after(0, lambda: self._load_error(str(e)))
    
    def _finish_load(self, df, path):
        self.drop_zone.browse_btn.configure(state="normal", text="Browse Files")
        self.app.set_dataframe(df, path)
        self.original_stats = {'rows': len(df), 'cols': len(df.columns), 'empty': sum(self._cached_empty.values()), 'dups': df.duplicated().sum()}
        self._columns_built = False
        self.update_ui(df, path)
    
    def _load_error(self, msg):
        self.drop_zone.browse_btn.configure(state="normal", text="Browse Files")
        messagebox.showerror("Error", msg)

    def update_ui(self, df, path):
        if df is None: return
        
        fname = os.path.basename(path)
        fsize = os.path.getsize(path)
        size_str = f"{fsize/1024:.1f} KB" if fsize < 1024*1024 else f"{fsize/(1024*1024):.1f} MB"
        
        self.file_label.configure(text=fname)
        self.file_size_label.configure(text=f"{size_str} • {len(df):,} rows × {len(df.columns)} cols")
        
        self.drop_zone.pack_forget()
        self.file_widget.pack(fill="x")
        
        total_empty = sum(self._cached_empty.values())
        dups = df.duplicated().sum()
        
        self.rows_lbl.configure(text=f"{len(df):,}")
        self.cols_lbl.configure(text=str(len(df.columns)))
        self.empty_lbl.configure(text=str(total_empty))
        self.dups_lbl.configure(text=str(dups))
        
        # Update empty values content
        if not self._cached_empty:
            self.empty_content.configure(text="✅ No empty values detected!")
        else:
            items = list(self._cached_empty.items())[:5]
            txt = " | ".join([f"{k[:12]}: {v}" for k, v in items])
            if len(self._cached_empty) > 5:
                txt += f" | +{len(self._cached_empty)-5} more"
            self.empty_content.configure(text=txt)
        
        self.preview_info.configure(text=f"Showing first 50 rows of {len(df):,}")
        
        self.status_frame.configure(fg_color="#ECFDF5")
        self.status_label.configure(text=f"✓ Loaded '{fname}' successfully", text_color="#059669")

    def clear_file(self):
        self.app.set_dataframe(None, None)
        self.file_widget.pack_forget()
        self.drop_zone.pack(fill="x")
        
        self.rows_lbl.configure(text="0")
        self.cols_lbl.configure(text="0")
        self.empty_lbl.configure(text="0")
        self.dups_lbl.configure(text="0")
        self.empty_content.configure(text="📊 Load data to see empty values")
        self.table_frame.populate_initial()
        self.preview_info.configure(text="")
        self.status_frame.configure(fg_color="#FEF3C7")
        self.status_label.configure(text="📁 No data loaded yet.", text_color="#92400E")
        self.steps_lbl.configure(text="")
        self._cached_empty = {}
        self._columns_built = False
        for w in self.col_content.winfo_children():
            w.destroy()
        self.column_options.clear()

    def apply_preprocessing(self):
        df = self.app.get_dataframe()
        if df is None:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        
        self.progress.pack(fill="x", pady=(0, 5))
        self.progress.start()
        self.apply_btn.configure(state="disabled", text="...")
        threading.Thread(target=self._process_thread, args=(df.copy(),), daemon=True).start()
    
    def _process_thread(self, df):
        steps = []
        try:
            # Replace placeholders with NaN
            for col in df.columns:
                if df[col].dtype == 'object':
                    mask = df[col].astype(str).str.strip().isin(EMPTY_VALUE_PLACEHOLDERS)
                    df.loc[mask, col] = np.nan
            
            one_hot = []
            for col, opts in self.column_options.items():
                if col not in df.columns: continue
                
                # Missing
                m = opts['miss'].get()
                if m != "None":
                    if m == "Drop":
                        b = len(df)
                        df = df.dropna(subset=[col])
                        if len(df) < b: steps.append(f"{col}: dropped {b-len(df)} rows")
                    elif m == "Mean" and opts['num']:
                        df[col] = df[col].fillna(df[col].mean())
                        steps.append(f"{col}: filled with mean")
                    elif m == "Median" and opts['num']:
                        df[col] = df[col].fillna(df[col].median())
                        steps.append(f"{col}: filled with median")
                    elif m == "Mode":
                        mode = df[col].mode()
                        if len(mode) > 0: df[col] = df[col].fillna(mode.iloc[0])
                        steps.append(f"{col}: filled with mode")
                    elif m == "FFill":
                        df[col] = df[col].ffill()
                        steps.append(f"{col}: forward filled")
                    elif m == "BFill":
                        df[col] = df[col].bfill()
                        steps.append(f"{col}: backward filled")
                
                # Scaling
                if opts['scale'] and opts['num']:
                    s = opts['scale'].get()
                    if s == "MinMax":
                        mn, mx = df[col].min(), df[col].max()
                        if mx != mn: df[col] = (df[col] - mn) / (mx - mn)
                        steps.append(f"{col}: MinMax scaled")
                    elif s == "ZScore":
                        mean, std = df[col].mean(), df[col].std()
                        if std != 0: df[col] = (df[col] - mean) / std
                        steps.append(f"{col}: Z-Score scaled")
                    elif s == "Robust":
                        med = df[col].median()
                        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                        iqr = q3 - q1
                        if iqr != 0: df[col] = (df[col] - med) / iqr
                        steps.append(f"{col}: Robust scaled")
                
                # Encoding
                if opts['encode'] and not opts['num']:
                    e = opts['encode'].get()
                    if e == "Label":
                        df[col] = pd.factorize(df[col])[0]
                        steps.append(f"{col}: label encoded")
                    elif e == "OneHot":
                        one_hot.append(col)
            
            if one_hot:
                df = pd.get_dummies(df, columns=one_hot)
                steps.append(f"One-hot: {', '.join(one_hot)}")
            
            # Duplicates
            d = self.dup_var.get()
            if d != "None":
                b = len(df)
                if d == "Remove All": df = df.drop_duplicates(keep=False)
                elif d == "Keep First": df = df.drop_duplicates(keep='first')
                elif d == "Keep Last": df = df.drop_duplicates(keep='last')
                if len(df) < b: steps.append(f"Removed {b-len(df)} duplicates")
            
            self.after(0, lambda: self._finish_processing(df, steps))
        except Exception as e:
            self.after(0, lambda: self._process_error(str(e)))
    
    def _finish_processing(self, df, steps):
        self.progress.stop()
        self.progress.pack_forget()
        self.apply_btn.configure(state="normal", text="Apply")
        
        self.app.set_dataframe(df, self.app.file_path)
        self._cached_empty = self._count_empty_fast(df)
        self._columns_built = False
        self.update_ui(df, self.app.file_path)
        self._build_column_options()
        
        if steps:
            self.steps_lbl.configure(text="✓ " + " | ".join(steps[:5]))
            self.steps_lbl.pack(fill="x", pady=(5, 0))
            messagebox.showinfo("Done", f"Applied {len(steps)} preprocessing steps!")
        else:
            messagebox.showinfo("No Changes", "No options selected.")
    
    def _process_error(self, msg):
        self.progress.stop()
        self.progress.pack_forget()
        self.apply_btn.configure(state="normal", text="Apply")
        messagebox.showerror("Error", msg)