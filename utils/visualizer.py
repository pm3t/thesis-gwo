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
    def plot_split_data(train_dates, train_values, test_dates, test_values, xlabel="date", ylabel="sales", ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
            ax.clear()

        # Plot training data
        ax.plot(train_dates, train_values, label=f'Data Latih (Train: {len(train_values)} sampel)', color='#1f77b4')
        # Plot testing data
        ax.plot(test_dates, test_values, label=f'Data Uji (Test: {len(test_values)} sampel)', color='#ff7f0e')

        # Add vertical line at the split boundary
        if len(train_dates) > 0:
            split_date = train_dates.iloc[-1] if hasattr(train_dates, 'iloc') else list(train_dates)[-1]
            ax.axvline(x=split_date, color='red', linestyle='--', label='Batas Split Data')

        ax.set_title("Visualisasi Split Data (Train vs Test)", fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=':', alpha=0.6)
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
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
            ax.clear()

        # Only display MA, ES, RNN, and GWO Ensemble
        display_models = ['MA', 'ES', 'LR', 'GWO Ensemble']
        models = [m for m in display_models if m in metrics_results]
        values = [metrics_results[m][metric_name] for m in models]

        # Define color map for the models
        color_map = {
            'MA': '#1f538d',
            'ES': '#3d609c',
            'LR': '#ff7f0e',
            'GWO Ensemble': '#2ca02c'
        }
        colors = [color_map.get(m, '#aaaaaa') for m in models]

        # Plot bars
        bars = ax.bar(models, values, color=colors, edgecolor='gray', linewidth=1, width=0.5)

        # Title and labels
        ax.set_title("Grafik perbandingan MAPE semua model", fontsize=14, fontweight="bold")
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("MAPE (%)" if metric_name == 'MAPE' else metric_name, fontsize=11)
        
        # Grid lines (horizontal only, dotted)
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.set_axisbelow(True) # Put grid below bars

        # Text labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            # Show percentage sign only if it's MAPE
            label_format = f'{height:.2f}%' if metric_name == 'MAPE' else f'{height:.4f}'
            ax.annotate(label_format,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        # Add y-axis padding to make space for annotations
        ax.margins(y=0.1)
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def plot_residuals(y_true, y_pred):
        """
        Plot residual analysis: residuals over time and distribution of residuals.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuals over time
        ax1.plot(residuals, color='red', alpha=0.7)
        ax1.axhline(0, color='black', linestyle='--')
        ax1.set_title("Residuals Over Time")
        ax1.set_xlabel("Time Index")
        ax1.set_ylabel("Residual (Actual - Predicted)")
        
        # Residual distribution
        sns.histplot(residuals, kde=True, ax=ax2, color='purple')
        ax2.set_title("Residuals Distribution")
        ax2.set_xlabel("Residual Value")
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_actual_vs_predicted(y_true, y_pred):
        """
        Plot scatter plot of Actual vs Predicted values.
        """
        from sklearn.metrics import r2_score
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_true, y_pred, alpha=0.7, color='blue', edgecolors='k')
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit (y = y_pred)')
        
        ax.set_title("Correlation: Actual vs Predicted")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.legend()
        
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig, ax
