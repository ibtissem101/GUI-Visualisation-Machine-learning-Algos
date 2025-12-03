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
        sidebar = ctk.CTkFrame(self, width=200, corner_radius=0, fg_color="#FAFAFA")
        sidebar.grid_propagate(False)
        
        # Logo
        logo_label = ctk.CTkLabel(
            sidebar, 
            text="Clustering Tool", 
            font=("Segoe UI", 16, "bold"),
            text_color="#1E293B",
            anchor="w"
        )
        logo_label.pack(padx=20, pady=(20, 0), anchor="w")
        
        subtitle = ctk.CTkLabel(
            sidebar,
            text="Data Science Platform",
            font=("Segoe UI", 11),
            text_color="#94A3B8",
            anchor="w"
        )
        subtitle.pack(padx=20, pady=(0, 20), anchor="w")
        
        # Menu
        menu_label = ctk.CTkLabel(
            sidebar,
            text="MAIN MENU",
            font=("Segoe UI", 10, "bold"),
            text_color="#94A3B8",
            anchor="w"
        )
        menu_label.pack(padx=20, pady=(10, 5), anchor="w")
        
        self.create_menu_btn(sidebar, "  Data Loader", "data_loader")
        self.create_menu_btn(sidebar, "  EDA", "eda")
        
        # Unsupervised Label
        ctk.CTkLabel(
            sidebar,
            text="▼ Unsupervised\n    Learning",
            text_color="#1E293B",
            anchor="w",
            font=("Segoe UI", 13),
            justify="left"
        ).pack(padx=10, pady=5, fill="x")
        
        # Submenu
        self.create_submenu_btn(sidebar, "K-Means", "kmeans")
        self.create_submenu_btn(sidebar, "K-Medoids", "kmedoids")
        self.create_submenu_btn(sidebar, "DIANA/AGNES", "hierarchical")
        self.create_submenu_btn(sidebar, "DBSCAN", "dbscan")
            
        self.create_menu_btn(sidebar, "📊  Visualization", "viz")
        
        # Bottom
        ctk.CTkFrame(sidebar, fg_color="transparent").pack(expand=True) # Spacer
        
        self.create_menu_btn(sidebar, "⚙️  Settings", "settings")
        self.create_menu_btn(sidebar, "❓  Help", "help")
        
        return sidebar

    def create_menu_btn(self, parent, text, page_name):
        btn = ctk.CTkButton(
            parent,
            text=text,
            fg_color="transparent",
            text_color="#1E293B",
            hover_color="#F1F5F9",
            anchor="w",
            font=("Segoe UI", 13),
            height=40,
            corner_radius=6,
            command=lambda: self.show_page(page_name)
        )
        btn.pack(padx=10, pady=2, fill="x")
        return btn

    def create_submenu_btn(self, parent, text, page_name):
        btn = ctk.CTkButton(
            parent,
            text=text,
            fg_color="transparent",
            text_color="#64748B",
            hover_color="#F1F5F9",
            anchor="w",
            font=("Segoe UI", 13),
            height=30,
            command=lambda: self.show_page(page_name)
        )
        btn.pack(padx=(30, 10), pady=2, fill="x")
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
        else:
            # Placeholder for unimplemented pages
            self.current_page = ctk.CTkLabel(self.content_container, text=f"Page '{page_name}' not implemented yet.")
            
        self.current_page.pack(fill="both", expand=True)

    def set_dataframe(self, df, file_path):
        self.df = df
        self.file_path = file_path
        
    def get_dataframe(self):
        return self.df