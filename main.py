"""
Main entry point for the PyQt6 GUI application.
"""
import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    """Initialize and run the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("PyQt6 Application")
    app.setOrganizationName("YourOrganization")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
