"""
Helper functions for the application.
"""
from tkinter import messagebox


def show_error(parent, title, message):
    """
    Show an error message dialog.
    
    Args:
        parent: Parent widget
        title: Dialog title
        message: Error message
    """
    messagebox.showerror(title, message, parent=parent)


def show_warning(parent, title, message):
    """
    Show a warning message dialog.
    
    Args:
        parent: Parent widget
        title: Dialog title
        message: Warning message
    """
    messagebox.showwarning(title, message, parent=parent)


def show_info(parent, title, message):
    """
    Show an information message dialog.
    
    Args:
        parent: Parent widget
        title: Dialog title
        message: Information message
    """
    messagebox.showinfo(title, message, parent=parent)


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
    return messagebox.askyesno(title, message, parent=parent)
