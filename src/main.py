import os
from data_loader import load_wisdm_data
from data_preprocessing import preprocess_data
from data_visualization import visualize_arff_data


def main():
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(current_dir, '..', 'wisdm-dataset'))
    print("Starting WISDM dataset analysis...")

    # Load data
    df = load_wisdm_data(data_dir, file_limit=20)

    if df is not None:
        # Preprocess the data
        segments, processed_df = preprocess_data(df, fs=20, window_duration=5, overlap=0.5)
        print(f"Processed DataFrame shape: {processed_df.shape}")
        print(f"Number of segments created: {len(segments)}")

        # Visualize the processed ARFF data
        visualize_arff_data(processed_df, show_plots=True)
        print("Data visualization complete. Check the 'visualizations/arff' folder for results.")
    else:
        print("Failed to load any data. Please check the data directory and file format.")


if __name__ == "__main__":
    main()
