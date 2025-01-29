# interface/input_mode.py

import tkinter as tk

def pick_input_mode():
    """
    Asks if user wants a single condition or multiple conditions.
    For multiple conditions, we will allow picking multiple folders.
    Returns:
      'single' or 'multiple'
    """
    root = tk.Tk()
    root.title("Select Input Mode")
    selection = []

    def choose_single():
        selection.append("single")
        root.destroy()

    def choose_multiple():
        selection.append("multiple")
        root.destroy()

    label = tk.Label(root, text="Do you want a single condition or multiple conditions?")
    label.pack(padx=20, pady=10)

    btn_single = tk.Button(root, text="Single Condition", command=choose_single)
    btn_single.pack(side=tk.LEFT, padx=20, pady=10)

    btn_multiple = tk.Button(root, text="Multiple Conditions", command=choose_multiple)
    btn_multiple.pack(side=tk.RIGHT, padx=20, pady=10)

    root.mainloop()

    if not selection:
        raise RuntimeError("No selection made.")
    return selection[0]
