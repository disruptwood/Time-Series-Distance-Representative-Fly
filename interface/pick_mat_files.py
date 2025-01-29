import os
from tkinter import filedialog, Tk

def pick_mat_files_single_condition():
    """
    Asks the user to pick multiple .mat files (all for one condition).
    Returns a list of absolute paths.
    """
    root = Tk()
    root.withdraw()
    chosen = filedialog.askopenfilenames(
        title='Select .mat Files (One Condition)',
        filetypes=[('MAT files', '*.mat')]
    )
    if not chosen:
        raise RuntimeError("No .mat files selected.")
    return [os.path.abspath(p) for p in chosen]
