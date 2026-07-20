import customtkinter as ctk
import sys
from ui.data_tab import DataTab
from ui.visualize_tab import VisualizeTab
from ui.model_tab import ModelTab
from ui.gwo_tab import GWOTab
from ui.ensemble_tab import EnsembleTab
from utils.data_processor import DataProcessor

# Default: light mode
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Optimasi Model Ensemble via GWO untuk Peramalan Penjualan")
        self.geometry("1200x800")
        
        # Protocol for closing window
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize Shared Data
        self.data_processor = DataProcessor()
        self.model_results = {} # Store results from ModelTab
        self.gwo_results = {}   # Store results from GWO

        # mode_sidang is the hidden logic behind the theme toggle
        self.mode_sidang = ctk.BooleanVar(value=False)
        
        # Configure layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        # Header Frame
        self.header_frame = ctk.CTkFrame(self)
        self.header_frame.grid(row=0, column=0, padx=20, pady=(15, 0), sticky="ew")

        self.lbl_title = ctk.CTkLabel(
            self.header_frame,
            text="Sistem Peramalan Penjualan - Optimasi GWO & Ensemble",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.lbl_title.pack(side="left", padx=20, pady=10)

        # Theme toggle (disguised Mode Sidang switch)
        self.theme_var = ctk.BooleanVar(value=False)
        self.switch_theme = ctk.CTkSwitch(
            self.header_frame,
            text="🌙  Dark Mode",
            variable=self.theme_var,
            command=self._on_theme_toggle,
            progress_color="#3b5998",
            button_color="#ffffff",
            button_hover_color="#dddddd"
        )
        self.switch_theme.pack(side="right", padx=20, pady=10)

        # Create Tabview
        self.tabview = ctk.CTkTabview(self, segmented_button_selected_color="#1f538d")
        self.tabview.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

        # Add Tabs
        self.tab_data = self.tabview.add("DATA")
        self.tab_visualize = self.tabview.add("VISUALISASI")
        self.tab_model = self.tabview.add("BASELINE MODEL")
        self.tab_gwo = self.tabview.add("OPTIMASI GWO")
        self.tab_ensemble = self.tabview.add("ENSEMBLE")

        # Initialize Tab Contents
        self.data_tab = DataTab(self.tab_data, self.data_processor)
        self.data_tab.pack(fill="both", expand=True)
        
        self.visualize_tab = VisualizeTab(self.tab_visualize, self.data_processor)
        self.visualize_tab.pack(fill="both", expand=True)
        
        self.model_tab = ModelTab(self.tab_model, self.data_processor, self.model_results, self.mode_sidang)
        self.model_tab.pack(fill="both", expand=True)
        
        self.gwo_tab = GWOTab(self.tab_gwo, self.data_processor, self.model_results, self.gwo_results, self.mode_sidang)
        self.gwo_tab.pack(fill="both", expand=True)
        
        self.ensemble_tab = EnsembleTab(self.tab_ensemble, self.data_processor, self.model_results, self.gwo_results, self.mode_sidang)
        self.ensemble_tab.pack(fill="both", expand=True)

    def _on_theme_toggle(self):
        """
        Toggle dark/light mode.
        Internally also activates/deactivates Mode Sidang.
        """
        is_dark = self.theme_var.get()
        if is_dark:
            ctk.set_appearance_mode("dark")
            self.switch_theme.configure(text="☀️  Light Mode")
            self.mode_sidang.set(True)
        else:
            ctk.set_appearance_mode("light")
            self.switch_theme.configure(text="🌙  Dark Mode")
            self.mode_sidang.set(False)

    def on_tab_change(self):
        current_tab = self.tabview.get()
        pass

    def on_closing(self):
        """
        Handle application closing.
        """
        self.quit()
        self.destroy()
        sys.exit(0)
