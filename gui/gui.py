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

        self.selected_file_label = None  # Initialize selected_file_label as None

        self.logo = PhotoImage(file="logo/sentetik_logo.png")
        self.root.iconphoto(True, self.logo)

        self.spin_var = tk.StringVar()
        self.spin_box_selection = ttk.Combobox(self.root, textvariable=self.spin_var, values=[], state="readonly")
        self.spin_box_selection.bind("<<ComboboxSelected>>", self.on_spin_select)

        self.next_button_selection = ttk.Button(self.root, text="Next", command=self.show_next_view, state="disabled")

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
        if selected_file:
            self.next_button_selection.config(state="active")
        else:
            self.next_button_selection.config(state="disabled")

    def create_selection_view(self):
        self.clear_view()
        
        self.view_labels[self.current_view_idx].pack(pady=10)
        
        self.spin_box_selection.pack(padx=10, pady=5)
        
        self.next_button_selection.config(text="Next", command=self.show_next_view)
        self.next_button_selection.pack(pady=10)

    def create_process_view(self):
        self.clear_view()
        self.view_labels[self.current_view_idx].pack(pady=10)

        # Get the selected file name from spin_var
        selected_file = self.spin_var.get()

        # Create a label to display the selected file name
        self.selected_file_label = tk.Label(self.root, text=f"Selected File: {selected_file}", font=("Helvetica", 12))
        self.selected_file_label.pack(pady=10)

        # Create radio buttons here
        self.create_radiobuttons()

        # Create the "Back" button and place it under the radio buttons
        self.back_button_process = ttk.Button(self.root, text="Back", command=self.show_selection_view)
        self.back_button_process.pack(side="left", padx=10, pady=10, anchor="sw")

        self.spin_box_selection.pack_forget()
        self.next_button_selection.pack_forget()


    def create_radiobuttons(self):
        # Create a radiobutton list
        self.process_radio_var = tk.StringVar()
        feature_selection_radio = ttk.Radiobutton(self.root, text="Feature Selection", variable=self.process_radio_var, value="feature_selection")
        machine_learning_radio = ttk.Radiobutton(self.root, text="Machine Learning", variable=self.process_radio_var, value="machine_learning")

        feature_selection_radio.pack(anchor="w", padx=10, pady=5)
        machine_learning_radio.pack(anchor="w", padx=10, pady=5)



    def show_selection_view(self):
        self.current_view_idx = 0
        self.clear_view()

        # Display the "Selection" label at the top
        self.view_labels[self.current_view_idx].pack(pady=10, anchor="n")

        # Remove the label that displays the selected file name if it exists
        if self.selected_file_label:
            self.selected_file_label.pack_forget()

        # Check if the back_button_process exists, and if it does, forget it to hide it
        if hasattr(self, 'back_button_process'):
            self.back_button_process.pack_forget()

        # Hide or destroy the radio buttons and their associated variable if they exist
        if hasattr(self, 'process_radio_var'):
            self.process_radio_var.set("")  # Unselect any radio button
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Radiobutton):
                    widget.pack_forget()
            delattr(self, 'process_radio_var')

        self.spin_box_selection.pack(padx=10, pady=5)

        self.next_button_selection.config(text="Next", command=self.show_next_view)
        self.next_button_selection.pack(pady=10)






    def clear_view(self):
        for label in self.view_labels:
            label.pack_forget()

if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()
