import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def preprocess_data(df, fs=20, window_duration=5, overlap=0.5, missing_threshold=0.5):
    """
    Preprocess sensor data by handling missing values,
    applying filters for noise reduction, normalizing, and
    segmenting data into fixed-size windows.

    Parameters:
        df (pd.DataFrame): sensor data.
        fs (int): sampling frequency (Hz).
        window_duration (int): window duration in seconds.
        overlap (float): overlap btw windows
        missing_threshold (float): max allowed fraction of missing values per segment

    Returns:
        segments (list of pd.DataFrame): list of segmented dataframes
        df_processed (pd.DataFrame): processed DataFrame
    """
    df_processed = df.copy()

    # 1. Handle missing values: input numeric columns with their mean
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    df_processed[numeric_cols] = df_processed[numeric_cols].apply(
        lambda col: pd.to_numeric(col, errors='coerce')
    )

    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())

    # 2. Noise Reduction: low-pass filter
    # nyq: Nyquist frequency
    def low_pass_filter(data, cutoff=3, fs=fs, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
        return filtfilt(b, a, data)

    sensor_cols = [col for col in numeric_cols if any(axis in col.lower() for axis in ['x', 'y', 'z'])]
    for col in sensor_cols:
        try:
            df_processed[col] = low_pass_filter(df_processed[col])
        except Exception as e:
            print(f"Warning: Could not filter column {col}: {e}")

    # 3. Z-score normalization
    for col in sensor_cols:
        mean_val = df_processed[col].mean()
        std_val = df_processed[col].std()
        if std_val != 0:
            df_processed[col] = (df_processed[col] - mean_val) / std_val
        else:
            df_processed[col] = df_processed[col] - mean_val

    # 4. Data Segmentation
    window_size = int(window_duration * fs)
    step_size = int(window_size * (1 - overlap))
    segments = []

    # Segment by index
    for start in range(0, len(df_processed) - window_size + 1, step_size):
        segment = df_processed.iloc[start: start + window_size]
        if segment.isna().mean().max() < missing_threshold:
            segments.append(segment)

    return segments, df_processed