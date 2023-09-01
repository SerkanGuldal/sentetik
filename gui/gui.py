import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
import os
from ttkthemes import ThemedStyle

class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("sentetik")

        style = ThemedStyle(self.root)
        style.set_theme("clam")

        self.views = ["Selection", "Process"]
        self.view_labels = []
        self.current_view_idx = 0

        self.file_list = tk.StringVar()
        self.selected_directory = "datasets"

        self.logo = PhotoImage(file="logo/sentetik_logo.png")
        self.root.iconphoto(True, self.logo)

        self.spin_var = tk.StringVar()
        self.spin_box_selection = ttk.Combobox(self.root, textvariable=self.spin_var, values=[], state="readonly")
        self.spin_box_selection.bind("<<ComboboxSelected>>", self.on_spin_select)

        self.next_button_selection = ttk.Button(self.root, text="Next", command=self.show_next_view)

        self.list_files()

        for view in self.views:
            label = tk.Label(root, text=view, font=("Helvetica", 14, "bold"))
            self.view_labels.append(label)

        self.create_selection_view()

    def show_next_view(self):
        self.current_view_idx = (self.current_view_idx + 1) % len(self.views)
        if self.current_view_idx == 0:
            self.create_selection_view()
        elif self.current_view_idx == 1:
            self.create_process_view()

    def list_files(self):
        files = os.listdir(self.selected_directory)
        self.file_list.set(" ".join(files))
        self.spin_box_selection["values"] = files

    def on_spin_select(self, event):
        selected_file = self.spin_var.get()
        print(f"Selected file: {selected_file}")

    def create_selection_view(self):
        self.clear_view()
        
        self.view_labels[self.current_view_idx].pack(pady=10)
        
        self.spin_box_selection.pack(padx=10, pady=5)
        
        self.next_button_selection.config(text="Next", command=self.show_next_view)
        self.next_button_selection.pack(pady=10)

    def create_process_view(self):
        self.clear_view()
        self.view_labels[self.current_view_idx].pack(pady=10)
        self.next_button_selection.config(text="Back", command=self.show_selection_view)
        self.next_button_selection.pack(side="right", padx=10, pady=10)
        self.spin_box_selection.pack_forget()  # Remove spin_box_selection

    def show_selection_view(self):
        self.current_view_idx = 0
        self.clear_view()

        self.view_labels[self.current_view_idx].pack(pady=10)
        
        self.spin_box_selection.pack(padx=10, pady=5)
        
        self.next_button_selection.config(text="Next", command=self.show_next_view)
        self.next_button_selection.pack(side="bottom", pady=10)



    def clear_view(self):
        for label in self.view_labels:
            label.pack_forget()

if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()
