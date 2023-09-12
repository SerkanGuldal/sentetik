import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
from tkinter import filedialog  # Import the filedialog module
import os
from ttkthemes import ThemedStyle
import shutil
import subprocess

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
        self.spin_box_selection = ttk.Combobox(self.root, textvariable=self.spin_var, values=[], state="readonly", width=30)
        self.spin_box_selection.bind("<<ComboboxSelected>>", self.on_spin_select)

        self.second_combobox_var = tk.StringVar()
        self.second_combobox = ttk.Combobox(self.root, textvariable=self.second_combobox_var, values=["Option 1", "Option 2"], state="readonly")

        self.next_button_selection = ttk.Button(self.root, text="Next", command=self.show_next_view, state="disabled")

        self.add_file_button_selection = ttk.Button(self.root, text="Add New Data", command=self.add_file)

        self.list_files()

        for view in self.views:
            label = tk.Label(root, text=view, font=("Helvetica", 14, "bold"))
            self.view_labels.append(label)

        self.create_selection_view()


    def add_file(self):
        # Open a file dialog to select CSV files
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV Files", "*.csv")])

        if file_paths:
            for file_path in file_paths:
                # Get the filename (without path) from the selected file
                file_name = os.path.basename(file_path)

                # Construct the destination path in the 'datasets' folder
                destination_path = os.path.join(self.selected_directory, file_name)

                # Copy the selected file to the 'datasets' folder
                try:
                    shutil.copy(file_path, destination_path)
                    # Refresh the list of CSV files in the Combobox
                    self.list_files()
                except Exception as e:
                    print(f"Error copying file: {str(e)}")


    def show_next_view(self):
        self.current_view_idx = (self.current_view_idx + 1) % len(self.views)
        if self.current_view_idx == 0:
            self.create_selection_view()
        elif self.current_view_idx == 1:
            self.create_process_view()
        elif self.current_view_idx == 3:
            self.create_feature_selection_view()

    def on_spin_select(self, event):
        selected_file = self.spin_var.get()
        print(f"Selected file: {selected_file}")
        if selected_file:
            self.next_button_selection.config(state="active")
        else:
            self.next_button_selection.config(state="disabled")

    def create_selection_view(self):
        self.clear_view()
        self.current_view_idx = 0

        self.view_labels[self.current_view_idx].grid(row=0, column=0, pady=10)

        # Call the modified list_files method to populate the Combobox
        self.list_files()
        self.spin_box_selection.grid(row=1, column=0, padx=10, pady=5)

        # Create and configure the "Add New Data" button
        self.add_file_button_selection.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.next_button_selection.config(text="Next", command=self.create_process_view)
        self.next_button_selection.grid(row=2, column=1, pady=10, padx=10)


    def list_files(self):
        # Create a list to store the CSV files
        csv_files = []

        # Walk through the selected directory and its subfolders
        for root_folder, subfolders, files in os.walk(self.selected_directory):
            for file in files:
                if file.endswith('.csv'):
                    # Remove the "datasets" folder name from the path
                    file_path = os.path.join(root_folder, file).replace(self.selected_directory + os.sep, "")
                    csv_files.append(file_path)

        # Calculate the desired width based on the longest file name
        if not csv_files:
            # Handle the case where no CSV files are found
            max_file_name_length = 30  # Set a default width of 30 characters
        else:
            max_file_name_length = max(len(file) for file in csv_files)

        desired_width = min(max_file_name_length, 300)  # Limit the width to a maximum of 100

        # Set the width of the Combobox
        self.spin_box_selection.configure(width=desired_width)

        # Convert the list of CSV files to a string and set it as the Combobox values
        self.spin_box_selection['values'] = csv_files



    def create_process_view(self):
        self.current_view_idx = 1
        self.clear_view()
        self.spin_box_selection.grid_forget()
        self.next_button_selection.grid_forget()

        self.view_labels[self.current_view_idx].grid(row=0, column=0, pady=10)

        selected_file = self.spin_var.get()
        self.selected_file_label = tk.Label(self.root, text=f"Selected File: {selected_file}", font=("Helvetica", 12))
        self.selected_file_label.grid(row=1, column=0, pady=10)

        # Call the process_list function to create the Combobox
        self.process_list()

        self.back_button_process = ttk.Button(self.root, text="Back", command=self.show_selection_view)
        self.back_button_process.grid(row=3, column=0, padx=10, pady=10, sticky="w")

        self.next_button_process = ttk.Button(self.root, text="Next", command=self.selected_process, state="disabled")
        self.next_button_process.grid(row=3, column=1, padx=10, pady=10, sticky="e")

        self.process_var.trace_add('write', self.check_radio_selection)

        self.spin_box_selection.grid_forget()
        self.next_button_selection.grid_forget()
        self.add_file_button_selection.grid_forget()



    def check_radio_selection(self, *args):
        selected_process = self.process_var.get()
        if selected_process:
            self.next_button_process.config(state="active")
        else:
            self.next_button_process.config(state="disabled")

    # Modify create_feature_selection_view method to store run_button as an instance variable
    def create_feature_selection_view(self):
        self.clear_view()
        self.process_combobox.grid_forget()
        
        if hasattr(self, 'selected_file_label'):
            self.selected_file_label.grid_forget()

        self.feature_selection_label = tk.Label(self.root, text="Feature Selection", font=("Helvetica", 14, "bold"))
        self.feature_selection_label.grid(row=0, column=0, pady=10)

        selected_file = self.spin_var.get()
        self.selected_file_label = tk.Label(self.root, text=f"Selected File:\n {selected_file}", font=("Helvetica", 12))
        self.selected_file_label.grid(row=1, column=0, pady=10)

        # Create the feature_selection_var and first Combobox
        self.feature_selection_var = tk.StringVar()
        self.feature_selection_combobox = ttk.Combobox(self.root, textvariable=self.feature_selection_var, values=["Chi2", "Second Method"])
        self.feature_selection_combobox.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        # Create the "Run" button and set its initial state to "disabled"
        self.run_button = ttk.Button(self.root, text="Run", command=lambda: self.run_feature_selection(selected_file))
        self.run_button.grid(row=3, column=1, padx=10, pady=10, sticky="e")
        self.run_button.configure(state="disabled")  # Disable the button initially

        # Create the "Back" button and set its initial state to "disabled"
        self.back_button_feature_selection = ttk.Button(self.root, text="Back", command=self.hide_feature_selection_view)
        self.back_button_feature_selection.grid(row=3, column=0, padx=10, pady=10, sticky="w")
   

        # Bind the event handler to the Combobox
        self.feature_selection_combobox.bind("<<ComboboxSelected>>", self.update_run_button_state)

    # Event handler to update the state of the "Run" button
    def update_run_button_state(self, event):
        selected_feature_method = self.feature_selection_var.get()
        
        if selected_feature_method:
            self.run_button.configure(state="active")  # Enable the button
        else:
            self.run_button.configure(state="disabled")  # Disable the button


    # Add this method to handle the "Run" button click
    def run_feature_selection(self, selected_file):
        selected_feature_method = self.feature_selection_var.get()

        if selected_feature_method == "Chi2":
            # Call the chi2.py script using subprocess with the selected file as input
            subprocess.run(["python", "featureSelection/chi2.py", selected_file])


        self.run_button.destroy()
        self.back_button_feature_selection.destroy()
        self.feature_selection_combobox.destroy()
        self.selected_file_label.destroy()
        self.feature_selection_label.destroy()

        # After running the feature selection, change the view to the selection view
        self.create_selection_view()



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


    def show_feature_selection_view(self):
        self.current_view_idx = 3
        self.create_feature_selection_view()

    def process_list(self):
            self.process_var = tk.StringVar()

            # Create a Combobox widget for process selection
            self.process_combobox = ttk.Combobox(self.root, textvariable=self.process_var, values=["Feature Selection", "Machine Learning"])
            self.process_combobox.grid(row=2, column=0, padx=10, pady=5, sticky="w")


    def show_selection_view(self):
        self.current_view_idx = 0
        self.clear_view()
        self.process_combobox.grid_forget()

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

        # Create and configure the "Add New Data" button
        self.add_file_button_selection = ttk.Button(self.root, text="Add New Data", command=self.add_file)
        self.add_file_button_selection.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        
        self.next_button_selection.config(text="Next", command=self.show_next_view)
        self.next_button_selection.grid(row=2, column=1, pady=10, padx=10)

        self.back_button_process.grid_forget()



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
