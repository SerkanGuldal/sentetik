from tkinter import Tk
from gui.gui import GUIApp  # Import the GUIApp class from gui.gui module

def main():
    root = Tk()
    app = GUIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
