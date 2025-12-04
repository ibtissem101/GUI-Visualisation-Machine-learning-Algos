import customtkinter as ctk
import tkinter as tk
import pandas as pd
import os

# Import pages
from ui.pages.data_loader import DataLoaderPage
from ui.pages.kmeans import KMeansPage
from ui.pages.kmedoids import KMedoidsPage
from ui.pages.hierarchical import HierarchicalPage
from ui.pages.dbscan import DBSCANPage
from ui.pages.comparison import ComparisonPage
from ui.pages.eda import EDAPage

# Set theme
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.df = None
        self.file_path = None
        self.current_page = None
        
        self.title("Clustering Tool")
        self.geometry("1380x800")
        self.configure(fg_color="#F5F5F5")
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Sidebar
        self.sidebar = self.create_sidebar()
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        # Content Container
        self.content_container = ctk.CTkFrame(self, fg_color="#F5F5F5", corner_radius=0)
        self.content_container.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        
        # Initial Page
        self.show_page("data_loader")
        
    def create_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color="#FFFFFF")
        sidebar.grid_propagate(False)
        
        # Header Section with Icon
        header_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 16))
        
        # Icon and title container
        title_container = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_container.pack(anchor="w")
        
        # Icon
        icon_label = ctk.CTkLabel(
            title_container,
            text="🔷",
            font=("Segoe UI", 20),
            anchor="w"
        )
        icon_label.pack(side="left", padx=(0, 8))
        
        # Title
        title_text_frame = ctk.CTkFrame(title_container, fg_color="transparent")
        title_text_frame.pack(side="left")
        
        logo_label = ctk.CTkLabel(
            title_text_frame, 
            text="Clustering Tool", 
            font=("Segoe UI", 16, "bold"),
            text_color="#1E293B",
            anchor="w"
        )
        logo_label.pack(anchor="w")
        
        subtitle = ctk.CTkLabel(
            title_text_frame,
            text="Data Science Platform",
            font=("Segoe UI", 10),
            text_color="#94A3B8",
            anchor="w"
        )
        subtitle.pack(anchor="w", pady=(2, 0))
        
        # Separator
        separator = ctk.CTkFrame(sidebar, height=1, fg_color="#E5E7EB")
        separator.pack(fill="x", padx=16, pady=(0, 12))
        
        # Main Menu Section
        menu_label = ctk.CTkLabel(
            sidebar,
            text="MAIN MENU",
            font=("Segoe UI", 9, "bold"),
            text_color="#94A3B8",
            anchor="w"
        )
        menu_label.pack(padx=16, pady=(4, 8), anchor="w")
        
        self.create_menu_btn(sidebar, "📁  Data Loader", "data_loader")
        self.create_menu_btn(sidebar, "📊  EDA", "eda")
        
        # Spacing
        ctk.CTkFrame(sidebar, height=4, fg_color="transparent").pack()
        
        # Unsupervised Section Container
        self.unsupervised_section = ctk.CTkFrame(sidebar, fg_color="transparent")
        self.unsupervised_section.pack(fill="x")

        # Collapsible Unsupervised Learning Section
        self.unsupervised_collapsed = ctk.BooleanVar(value=True)
        
        unsupervised_header = ctk.CTkFrame(self.unsupervised_section, fg_color="transparent")
        unsupervised_header.pack(fill="x", padx=8, pady=4)
        
        self.collapse_icon = ctk.CTkLabel(
            unsupervised_header,
            text="▶",
            font=("Segoe UI", 10),
            text_color="#64748B",
            width=20
        )
        self.collapse_icon.pack(side="left", padx=(8, 4))
        
        unsupervised_btn = ctk.CTkButton(
            unsupervised_header,
            text="Unsupervised Learning",
            fg_color="transparent",
            text_color="#1E293B",
            hover_color="#F1F5F9",
            anchor="w",
            font=("Segoe UI", 13),
            height=36,
            corner_radius=6,
            command=self.toggle_unsupervised
        )
        unsupervised_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))
        
        # Submenu Container
        self.submenu_container = ctk.CTkFrame(self.unsupervised_section, fg_color="transparent")
        # Initially hidden since we start collapsed
        
        self.create_submenu_btn(self.submenu_container, "K-Means", "kmeans")
        self.create_submenu_btn(self.submenu_container, "K-Medoids", "kmedoids")
        self.create_submenu_btn(self.submenu_container, "DIANA/AGNES", "hierarchical")
        self.create_submenu_btn(self.submenu_container, "DBSCAN", "dbscan")
        self.create_submenu_btn(self.submenu_container, "Comparison", "comparison")
        
        # Spacing
        ctk.CTkFrame(sidebar, height=4, fg_color="transparent").pack()
        
        # Spacer to push bottom items down
        ctk.CTkFrame(sidebar, fg_color="transparent").pack(expand=True)
        
        # Bottom separator
        separator_bottom = ctk.CTkFrame(sidebar, height=1, fg_color="#E5E7EB")
        separator_bottom.pack(fill="x", padx=16, pady=(0, 12))
        
        # Bottom Menu Items
        self.create_menu_btn(sidebar, "⚙️  Settings", "settings")
        self.create_menu_btn(sidebar, "❓  Help", "help")
        
        # Bottom padding
        ctk.CTkFrame(sidebar, height=16, fg_color="transparent").pack()
        
        return sidebar
    
    def toggle_unsupervised(self):
        is_collapsed = self.unsupervised_collapsed.get()
        
        if is_collapsed:
            # Expand
            self.submenu_container.pack(fill="x", pady=(0, 4))
            self.collapse_icon.configure(text="▼")
            self.unsupervised_collapsed.set(False)
        else:
            # Collapse
            self.submenu_container.pack_forget()
            self.collapse_icon.configure(text="▶")
            self.unsupervised_collapsed.set(True)

    def create_menu_btn(self, parent, text, page_name):
        btn = ctk.CTkButton(
            parent,
            text=text,
            fg_color="transparent",
            text_color="#1E293B",
            hover_color="#F1F5F9",
            anchor="w",
            font=("Segoe UI", 13),
            height=42,
            corner_radius=8,
            command=lambda: self.show_page(page_name)
        )
        btn.pack(padx=12, pady=3, fill="x")
        return btn

    def create_submenu_btn(self, parent, text, page_name):
        btn = ctk.CTkButton(
            parent,
            text=text,
            fg_color="transparent",
            text_color="#64748B",
            hover_color="#F1F5F9",
            anchor="w",
            font=("Segoe UI", 12),
            height=36,
            corner_radius=6,
            command=lambda: self.show_page(page_name)
        )
        btn.pack(padx=(32, 12), pady=2, fill="x")
        return btn

    def show_page(self, page_name):
        # Clear current page
        for widget in self.content_container.winfo_children():
            widget.destroy()
            
        if page_name == "data_loader":
            self.current_page = DataLoaderPage(self.content_container, self)
        elif page_name == "eda":
            self.current_page = EDAPage(self.content_container, self)
        elif page_name == "kmeans":
            self.current_page = KMeansPage(self.content_container, self)
        elif page_name == "kmedoids":
            self.current_page = KMedoidsPage(self.content_container, self)
        elif page_name == "hierarchical":
            self.current_page = HierarchicalPage(self.content_container, self)
        elif page_name == "dbscan":
            self.current_page = DBSCANPage(self.content_container, self)
        elif page_name == "comparison":
            self.current_page = ComparisonPage(self.content_container, self)
        else:
            # Placeholder for unimplemented pages
            self.current_page = ctk.CTkLabel(
                self.content_container, 
                text=f"Page '{page_name}' not implemented yet.",
                font=("Segoe UI", 14),
                text_color="#64748B"
            )
            
        self.current_page.pack(fill="both", expand=True)
        
        # Update feature options if page has that method (after brief delay for UI to settle)
        if hasattr(self.current_page, 'update_feature_options'):
            self.after(100, self.current_page.update_feature_options)

    def set_dataframe(self, df, file_path):
        self.df = df
        self.file_path = file_path
        
    def get_dataframe(self):
        return self.df

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()