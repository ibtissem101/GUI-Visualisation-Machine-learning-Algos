"""
Helper functions for the application.
"""
from PyQt6.QtWidgets import QMessageBox


def show_error(parent, title, message):
    """
    Show an error message dialog.
    
    Args:
        parent: Parent widget
        title: Dialog title
        message: Error message
    """
    QMessageBox.critical(parent, title, message)


def show_warning(parent, title, message):
    """
    Show a warning message dialog.
    
    Args:
        parent: Parent widget
        title: Dialog title
        message: Warning message
    """
    QMessageBox.warning(parent, title, message)


def show_info(parent, title, message):
    """
    Show an information message dialog.
    
    Args:
        parent: Parent widget
        title: Dialog title
        message: Information message
    """
    QMessageBox.information(parent, title, message)


def ask_question(parent, title, message):
    """
    Show a yes/no question dialog.
    
    Args:
        parent: Parent widget
        title: Dialog title
        message: Question message
        
    Returns:
        bool: True if user clicked Yes, False otherwise
    """
    reply = QMessageBox.question(
        parent,
        title,
        message,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    return reply == QMessageBox.StandardButton.Yes
