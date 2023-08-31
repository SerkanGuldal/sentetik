from tkinter import Tk
from gui import GUIApp  # Import the GUIApp class from gui_app module

def main():
    root = Tk()
    app = GUIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
