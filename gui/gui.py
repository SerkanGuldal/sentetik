import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
import os
from ttkthemes import ThemedStyle

class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("sentetik")

        # Calculate the screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Calculate the x and y position for the centered window
        x = (screen_width - 800) // 2  # Set the width of your window here (e.g., 800)
        y = (screen_height - 600) // 2  # Set the height of your window here (e.g., 600)

        # Use the geometry method to set the position without resizing
        self.root.geometry(f"+{x}+{y}")

        style = ThemedStyle(self.root)
        style.set_theme("clam")

        self.views = ["Selection", "Process", "ML", "FeatureSelection"]
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

        self.feature_selection_var = tk.StringVar()
        self.feature_selection_combobox = ttk.Combobox(self.root, textvariable=self.feature_selection_var, values=["Chi2", "Second Method"], state="readonly")

        self.second_combobox_var = tk.StringVar()
        self.second_combobox = ttk.Combobox(self.root, textvariable=self.second_combobox_var, values=["Option 1", "Option 2"], state="readonly")

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
        elif self.current_view_idx == 3:
            self.create_feature_selection_view()

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
        self.view_labels[self.current_view_idx].grid(row=0, column=0, pady=10)

        self.spin_box_selection.grid(row=1, column=0, padx=10, pady=5)

        self.next_button_selection.config(text="Next", command=self.show_next_view)
        self.next_button_selection.grid(row=2, column=0, pady=10)

    def create_process_view(self):
        self.clear_view()
        self.view_labels[self.current_view_idx].grid(row=0, column=0, pady=10)

        selected_file = self.spin_var.get()
        self.selected_file_label = tk.Label(self.root, text=f"Selected File: {selected_file}", font=("Helvetica", 12))
        self.selected_file_label.grid(row=1, column=0, pady=10)

        # Call the process_list function to create the Combobox
        self.process_list()

        self.back_button_process = ttk.Button(self.root, text="Back", command=self.show_selection_view)
        self.back_button_process.grid(row=3, column=0, padx=10, pady=10, sticky="sw")  # Move Back button to row 3

        self.next_button_process = ttk.Button(self.root, text="Next", command=self.selected_process, state="disabled")
        self.next_button_process.grid(row=3, column=1, padx=10, pady=10, sticky="se")  # Move Next button to row 3

        self.process_var.trace_add('write', self.check_radio_selection)

        self.spin_box_selection.grid_forget()
        self.next_button_selection.grid_forget()

    def check_radio_selection(self, *args):
        selected_process = self.process_var.get()
        if selected_process:
            self.next_button_process.config(state="active")
        else:
            self.next_button_process.config(state="disabled")

    def create_feature_selection_view(self):
        self.clear_view()
        if hasattr(self, 'selected_file_label'):
            self.selected_file_label.grid_forget()

        feature_selection_label = tk.Label(self.root, text="Feature Selection", font=("Helvetica", 14, "bold"))
        feature_selection_label.grid(row=0, column=0, pady=10)

        selected_file = self.spin_var.get()
        selected_file_label = tk.Label(self.root, text=f"Selected File: {selected_file}", font=("Helvetica", 12))
        selected_file_label.grid(row=1, column=0, pady=10)

        # Create the feature_selection_var and first Combobox
        self.feature_selection_var = tk.StringVar()
        feature_selection_combobox = ttk.Combobox(self.root, textvariable=self.feature_selection_var, values=["Chi2", "Second Method"])
        feature_selection_combobox.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        back_button_feature_selection = ttk.Button(self.root, text="Back", command=self.hide_feature_selection_view)
        back_button_feature_selection.grid(row=3, column=0, padx=10, pady=10, sticky="w")

        # Add the "Run" button at the bottom right
        run_button = ttk.Button(self.root, text="Run")
        run_button.grid(row=3, column=1, padx=10, pady=10, sticky="e")

        self.spin_box_selection.grid_forget()
        self.next_button_selection.grid_forget()

    def hide_feature_selection_view(self):
        for widget in self.root.winfo_children():
            if widget != self.next_button_process:
                widget.grid_forget()
        self.show_process_view()

    def selected_process(self):
        selected_process = self.process_var.get()

        if selected_process == "Machine Learning":  # Adjust the comparison based on Combobox values
            self.show_ml_view()
        elif selected_process == "Feature Selection":  # Adjust the comparison based on Combobox values
            self.show_feature_selection_view()
        else:
            self.show_next_view()

        if selected_process == "Feature Selection":  # Adjust the comparison based on Combobox values
            self.back_button_process.grid_forget()
            self.next_button_process.grid_forget()
        else:
            self.back_button_process.grid(row=2, column=0, padx=10, pady=10, sticky="sw")
            self.next_button_process.grid(row=2, column=0, padx=10, pady=10, sticky="se")

    def show_ml_view(self):
        self.current_view_idx = 2
        self.create_ml_view()

    def create_ml_view(self):
        self.clear_view()

        ml_label = tk.Label(self.root, text="Machine Learning View", font=("Helvetica", 14, "bold"))
        ml_label.grid(row=1, column=0)

        self.back_button_ml = ttk.Button(self.root, text="Back", command=self.show_process_view)
        self.back_button_ml.grid(row=2, column=0, padx=10, pady=10, sticky="sw")

        self.next_button_ml = ttk.Button(self.root, text="Next", command=self.show_selection_view)
        self.next_button_ml.grid(row=2, column=1, padx=10, pady=10, sticky="se")

        self.next_button_selection.grid_forget()
        self.spin_box_selection.grid_forget()
        self.process_combobox.grid_foraget()

    def show_feature_selection_view(self):
        self.current_view_idx = 3
        self.create_feature_selection_view()

    def process_list(self):
            self.process_var = tk.StringVar()

            # Create a Combobox widget for process selection
            process_combobox = ttk.Combobox(self.root, textvariable=self.process_var, values=["Feature Selection", "Machine Learning"])
            process_combobox.grid(row=2, column=0, padx=10, pady=5, sticky="w")


    def show_selection_view(self):
        self.current_view_idx = 0
        self.clear_view()
        self.view_labels[self.current_view_idx].grid(row=0, column=0, pady=10, sticky="n")

        if hasattr(self, 'selected_file_label'):
            self.selected_file_label.grid_forget()

        if hasattr(self, 'back_button_process'):
            self.back_button_process.grid_forget()

        if hasattr(self, 'next_button_process'):
            self.next_button_process.destroy()

        if hasattr(self, 'process_var'):
            self.process_var.set("")
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Radiobutton):
                    widget.grid_forget()
            delattr(self, 'process_var')

        self.spin_box_selection.grid(row=1, column=0, padx=10, pady=5)

        self.next_button_selection.config(text="Next", command=self.show_next_view)
        self.next_button_selection.grid(row=2, column=0, pady=10)

    def show_process_view(self):
        self.current_view_idx = 1
        self.clear_view()
        self.view_labels[self.current_view_idx].grid(row=0, column=0, pady=10)

        selected_file = self.spin_var.get()
        self.selected_file_label = tk.Label(self.root, text=f"Selected File: {selected_file}", font=("Helvetica", 12))
        self.selected_file_label.grid(row=1, column=0, pady=10)

        self.process_list()

        self.back_button_process = ttk.Button(self.root, text="Back", command=self.show_selection_view)
        self.back_button_process.grid(row=3, column=0, padx=10, pady=10, sticky="sw")

        self.next_button_process = ttk.Button(self.root, text="Next", command=self.selected_process, state="disabled")
        self.next_button_process.grid(row=3, column=1, padx=10, pady=10, sticky="se")

        self.process_var.trace_add('write', self.check_radio_selection)

        self.spin_box_selection.grid_forget()
        self.next_button_selection.grid_forget()

    def clear_view(self):
        for label in self.view_labels:
            label.grid_forget()

if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()
