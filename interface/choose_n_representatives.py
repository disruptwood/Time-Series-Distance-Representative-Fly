import tkinter as tk
from tkinter import simpledialog

def pick_n_representatives():
    """
    Ask user for an integer n: how many representative flies to select.
    """
    root = tk.Tk()
    root.withdraw()
    val = simpledialog.askinteger("N Representatives", "Enter how many flies to pick:", minvalue=1)
    if val is None:
        raise RuntimeError("No valid n provided.")
    return val
