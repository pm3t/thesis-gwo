import customtkinter as ctk
import sys
from ui.main_window import MainWindow

def main():
    # Set appearance mode and color theme
    ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
    ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

    app = MainWindow()
    app.mainloop()
    sys.exit(0)

if __name__ == "__main__":
    main()
