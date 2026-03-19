import customtkinter as ctk
import sys
from ui.data_tab import DataTab
from ui.visualize_tab import VisualizeTab
from ui.model_tab import ModelTab
from ui.gwo_tab import GWOTab
from ui.ensemble_tab import EnsembleTab
from utils.data_processor import DataProcessor

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
        
        # Configure layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create Tabview
        self.tabview = ctk.CTkTabview(self, segmented_button_selected_color="#1f538d")
        self.tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

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
        
        self.model_tab = ModelTab(self.tab_model, self.data_processor, self.model_results)
        self.model_tab.pack(fill="both", expand=True)
        
        self.gwo_tab = GWOTab(self.tab_gwo, self.data_processor, self.model_results, self.gwo_results)
        self.gwo_tab.pack(fill="both", expand=True)
        
        self.ensemble_tab = EnsembleTab(self.tab_ensemble, self.data_processor, self.model_results, self.gwo_results)
        self.ensemble_tab.pack(fill="both", expand=True)

        # Setup Tab Change Listener if needed
        # self.tabview.configure(command=self.on_tab_change)

    def on_tab_change(self):
        current_tab = self.tabview.get()
        # You can add logic here to refresh tabs when switched
        pass

    def on_closing(self):
        """
        Handle application closing.
        """
        self.quit()
        self.destroy()
        sys.exit(0)
