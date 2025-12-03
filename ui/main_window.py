"""
Main window for the PyQt6 application.
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QMenuBar,
    QStatusBar, QToolBar
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 Application")
        self.setMinimumSize(800, 600)
        
        # Initialize UI components
        self._create_menu_bar()
        self._create_toolbar()
        self._create_central_widget()
        self._create_status_bar()
        
    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.on_new)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.on_open)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        copy_action = QAction("&Copy", self)
        copy_action.setShortcut("Ctrl+C")
        edit_menu.addAction(copy_action)
        
        paste_action = QAction("&Paste", self)
        paste_action.setShortcut("Ctrl+V")
        edit_menu.addAction(paste_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)
        
    def _create_toolbar(self):
        """Create the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        
        # Add actions to toolbar
        new_button = QAction("New", self)
        new_button.triggered.connect(self.on_new)
        toolbar.addAction(new_button)
        
        open_button = QAction("Open", self)
        open_button.triggered.connect(self.on_open)
        toolbar.addAction(open_button)
        
        toolbar.addSeparator()
        
        save_button = QAction("Save", self)
        save_button.triggered.connect(self.on_save)
        toolbar.addAction(save_button)
        
    def _create_central_widget(self):
        """Create the central widget with main content."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Title label
        title = QLabel("Welcome to PyQt6 Application")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 20px;")
        layout.addWidget(title)
        
        # Input section
        input_layout = QHBoxLayout()
        input_label = QLabel("Enter text:")
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type something here...")
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_field)
        layout.addLayout(input_layout)
        
        # Button section
        button_layout = QHBoxLayout()
        
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.on_submit)
        button_layout.addWidget(self.submit_button)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.on_clear)
        button_layout.addWidget(self.clear_button)
        
        layout.addLayout(button_layout)
        
        # Output area
        output_label = QLabel("Output:")
        layout.addWidget(output_label)
        
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setPlaceholderText("Output will appear here...")
        layout.addWidget(self.output_area)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    # Event handlers
    def on_new(self):
        """Handle New action."""
        self.status_bar.showMessage("New file created")
        self.output_area.clear()
        self.input_field.clear()
        
    def on_open(self):
        """Handle Open action."""
        self.status_bar.showMessage("Open file dialog would appear here")
        
    def on_save(self):
        """Handle Save action."""
        self.status_bar.showMessage("Save file dialog would appear here")
        
    def on_about(self):
        """Handle About action."""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "About PyQt6 Application",
            "This is a sample PyQt6 application.\n\n"
            "Built with PyQt6 framework for Python."
        )
        
    def on_submit(self):
        """Handle Submit button click."""
        text = self.input_field.text()
        if text:
            self.output_area.append(f"You entered: {text}")
            self.status_bar.showMessage(f"Processed: {text}")
        else:
            self.status_bar.showMessage("Please enter some text first", 3000)
            
    def on_clear(self):
        """Handle Clear button click."""
        self.input_field.clear()
        self.output_area.clear()
        self.status_bar.showMessage("Cleared")
