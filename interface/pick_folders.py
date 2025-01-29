# interface/pick_folders.py

import os
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import messagebox

def pick_multiple_folders_for_conditions():
    """
    Opens a window where users can drag and drop multiple folders.
    Returns a list of absolute folder paths.
    Raises a RuntimeError if no valid folders are dropped.
    """
    folders = []

    def on_drop(event):
        nonlocal folders
        # event.data can contain multiple paths separated by space or curly braces
        raw_paths = event.data
        # Handle different OS path separators and formats
        if raw_paths.startswith('{') and raw_paths.endswith('}'):
            # Multiple paths are enclosed in {}
            raw_paths = raw_paths[1:-1]
            paths = raw_paths.split('} {')
        else:
            paths = raw_paths.split()

        # Clean and filter directories
        for path in paths:
            path = path.strip('{}')  # Remove any surrounding braces
            path = os.path.abspath(path)
            if os.path.isdir(path):
                folders.append(path)

        # Close the window after dropping
        root.quit()

    root = TkinterDnD.Tk()
    root.title("Drag and Drop Folders")
    root.geometry("400x200")

    label = tk.Label(root, text="Drag and drop folders here", padx=10, pady=10)
    label.pack(expand=True, fill='both')

    # Make the label a drop target
    label.drop_target_register(DND_FILES)
    label.dnd_bind('<<Drop>>', on_drop)

    root.mainloop()

    if not folders:
        raise RuntimeError("No valid folders were dropped.")

    return folders
