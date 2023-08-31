import tkinter as tk

class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Navigation App")
        
        self.views = ["View 1", "View 2", "View 3"]  # Replace with your content
        
        self.current_view_idx = 0
        self.label = tk.Label(root, text=self.views[self.current_view_idx], padx=10, pady=10)
        self.label.pack()

        self.back_button = tk.Button(root, text="Back", command=self.show_previous_view)
        self.back_button.pack(side="left", padx=10, pady=10)

        self.forward_button = tk.Button(root, text="Forward", command=self.show_next_view)
        self.forward_button.pack(side="right", padx=10, pady=10)

    def show_previous_view(self):
        self.current_view_idx = (self.current_view_idx - 1) % len(self.views)
        self.label.config(text=self.views[self.current_view_idx])

    def show_next_view(self):
        self.current_view_idx = (self.current_view_idx + 1) % len(self.views)
        self.label.config(text=self.views[self.current_view_idx])

if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()
