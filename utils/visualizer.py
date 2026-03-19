import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Visualizer:
    @staticmethod
    def plot_time_series(dates, values, title="Time Series Data", xlabel="Date", ylabel="Sales", ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
            ax.clear()

        ax.plot(dates, values, label='Actual Data', color='blue')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_decomposition(df, date_col, target_col):
        # Set date as index and ensure frequency
        temp_df = df.set_index(date_col)
        # Assuming daily data, but it might have gaps. For decomposition, we need regular interval.
        # Simple re-indexing to handle gaps
        temp_df = temp_df[target_col].resample('D').mean().interpolate()
        
        result = seasonal_decompose(temp_df, model='additive', period=30) # period 30 for monthly seasonality
        
        fig = result.plot()
        fig.set_size_inches(10, 8)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_distribution(values, title="Data Distribution", ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
            ax.clear()

        sns.histplot(values, kde=True, ax=ax, color='green')
        ax.set_title(title)
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_convergence(curve, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
            ax.clear()

        ax.plot(range(1, len(curve) + 1), curve, color='red', marker='o', markersize=2)
        ax.set_title("GWO Convergence Curve")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness (MAPE)")
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_comparison(dates, y_actual, predictions_dict, title="Model Predictions Comparison", ax=None):
        """
        predictions_dict: { 'Model Name': y_pred_values }
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure
            ax.clear()

        ax.plot(dates, y_actual, label='Actual', color='black', alpha=0.7, linewidth=2)
        
        for name, pred in predictions_dict.items():
            ax.plot(dates, pred, label=name, alpha=0.8)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_error_comparison(metrics_results, metric_name='MAPE', ax=None):
        """
        metrics_results: { 'MA': { 'MAPE': 10, ... }, 'ES': { ... } }
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
            ax.clear()

        models = list(metrics_results.keys())
        values = [metrics_results[m][metric_name] for m in models]

        sns.barplot(x=models, y=values, ax=ax)
        ax.set_title(f"{metric_name} Comparison Across Models")
        ax.set_ylabel(metric_name)
        plt.tight_layout()
        return fig, ax
