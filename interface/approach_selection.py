import tkinter as tk

def pick_approach():
    """
    Asks the user to pick one of four approaches (1..4).
    Returns an integer 1..4.
    """
    root = tk.Tk()
    root.title("Select Approach")
    selection = []

    def choose_1():
        selection.append(1)
        root.destroy()

    def choose_2():
        selection.append(2)
        root.destroy()

    def choose_3():
        selection.append(3)
        root.destroy()

    def choose_4():
        selection.append(4)
        root.destroy()

    label = tk.Label(root, text="Choose an approach:")
    label.pack(padx=20, pady=10)

    btn1 = tk.Button(root, text="Approach 1: Multi-Hot Weighted Hamming", command=choose_1)
    btn1.pack(padx=10, pady=5, fill="x")
    btn2 = tk.Button(root, text="Approach 2: Interval Overlap Distance", command=choose_2)
    btn2.pack(padx=10, pady=5, fill="x")
    btn3 = tk.Button(root, text="Approach 3: Multi-Hot Markov Model", command=choose_3)
    btn3.pack(padx=10, pady=5, fill="x")
    btn4 = tk.Button(root, text="Approach 4: Numeric Scores + Euclidean/DTW", command=choose_4)
    btn4.pack(padx=10, pady=5, fill="x")

    root.mainloop()
    if not selection:
        raise RuntimeError("No approach selected.")
    return selection[0]
