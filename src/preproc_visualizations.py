import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_wisdm_data
from data_preprocessing import preprocess_data


def get_sensor_columns(df):
    """Return numeric columns whose names indicate sensor readings (contain x, y, or z)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sensor_cols = [col for col in numeric_cols if any(axis in col.lower() for axis in ['x', 'y', 'z'])]
    return sensor_cols


def make_safe_filename(name):
    """Remove quotes and spaces from a string to create a safe filename."""
    return name.replace('"', '').replace("'", "").replace(" ", "_")


def plot_histogram_comparison(raw_df, processed_df, sensor_col, output_dir):
    """Plot histograms for a given sensor column with the same scale on both plots."""
    # Get the combined min and max to use the same binning
    min_val = min(raw_df[sensor_col].min(), processed_df[sensor_col].min())
    max_val = max(raw_df[sensor_col].max(), processed_df[sensor_col].max())
    bins = np.linspace(min_val, max_val, 50)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    axs[0].hist(raw_df[sensor_col].dropna(), bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    axs[0].set_title(f"Raw Data: {sensor_col}")
    axs[0].set_xlabel(sensor_col)
    axs[0].set_ylabel("Frequency")

    axs[1].hist(processed_df[sensor_col].dropna(), bins=bins, alpha=0.7, color='lightgreen', edgecolor='black')
    axs[1].set_title(f"Processed Data: {sensor_col}")
    axs[1].set_xlabel(sensor_col)

    plt.suptitle(f"Histogram Comparison for {sensor_col}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    safe_sensor = make_safe_filename(sensor_col)
    fig_path = os.path.join(output_dir, f"histogram_comparison_{safe_sensor}.png")
    plt.savefig(fig_path)
    plt.close()


def plot_boxplot_comparison(raw_df, processed_df, sensor_col, output_dir):
    """Plot boxplots for a sensor column using the same y-axis scale."""
    # Combine the raw and processed data into one DataFrame
    combined_data = pd.DataFrame({
        "Raw": raw_df[sensor_col],
        "Processed": processed_df[sensor_col]
    })

    # Determine common y-axis limits from the combined data
    y_min = combined_data.min().min()
    y_max = combined_data.max().max()

    fig, ax = plt.subplots(figsize=(8, 6))
    combined_data.boxplot(ax=ax)
    ax.set_title(f"Boxplot Comparison for {sensor_col}", fontsize=14)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel(sensor_col)

    safe_sensor = make_safe_filename(sensor_col)
    fig_path = os.path.join(output_dir, f"boxplot_comparison_{safe_sensor}.png")
    plt.savefig(fig_path)
    plt.close()


def plot_timeseries_comparison(raw_df, processed_df, sensor_col, output_dir, num_points=200):
    """Plot the first few time series points for direct comparison with the same y-scale."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x_values = np.arange(num_points)

    raw_series = raw_df[sensor_col].iloc[:num_points]
    processed_series = processed_df[sensor_col].iloc[:num_points]

    axs[0].plot(x_values, raw_series, color='skyblue')
    axs[0].set_title(f"Raw Data (First {num_points} Points): {sensor_col}")
    axs[0].set_xlabel("Index")
    axs[0].set_ylabel(sensor_col)

    axs[1].plot(x_values, processed_series, color='lightgreen')
    axs[1].set_title(f"Processed Data (First {num_points} Points): {sensor_col}")
    axs[1].set_xlabel("Index")

    # Ensure the same y-limits for both subplots
    y_min = min(raw_series.min(), processed_series.min())
    y_max = max(raw_series.max(), processed_series.max())
    axs[0].set_ylim(y_min, y_max)
    axs[1].set_ylim(y_min, y_max)

    plt.suptitle(f"Time Series Comparison for {sensor_col}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    safe_sensor = make_safe_filename(sensor_col)
    fig_path = os.path.join(output_dir, f"timeseries_comparison_{safe_sensor}.png")
    plt.savefig(fig_path)
    plt.close()


def main():
    # Define directories
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(current_dir, '..', 'wisdm-dataset'))
    comparison_vis_dir = os.path.abspath(os.path.join(current_dir, 'comparison'))
    if not os.path.exists(comparison_vis_dir):
        os.makedirs(comparison_vis_dir)

    print("Loading raw WISDM dataset...")
    raw_df = load_wisdm_data(data_dir, file_limit=20)
    if raw_df is None:
        print("No data loaded. Please check your data directory and file format.")
        return

    print("Preprocessing data...")
    segments, processed_df = preprocess_data(raw_df, fs=20, window_duration=5, overlap=0.5)

    # Choose a sensor column to compare (for example, the first sensor column)
    sensor_cols = get_sensor_columns(raw_df)
    if not sensor_cols:
        print("No sensor columns found in the dataset.")
        return
    sensor_col = sensor_cols[0]
    print(f"Creating comparison visualizations for sensor column: {sensor_col}")

    # Create and save comparison visualizations
    plot_histogram_comparison(raw_df, processed_df, sensor_col, comparison_vis_dir)
    plot_boxplot_comparison(raw_df, processed_df, sensor_col, comparison_vis_dir)
    plot_timeseries_comparison(raw_df, processed_df, sensor_col, comparison_vis_dir)

    print("Comparison visualizations have been saved to:", comparison_vis_dir)


if __name__ == "__main__":
    main()
