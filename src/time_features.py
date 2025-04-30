import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def load_raw_data(data_dir, file_limit=None):
    """
    Load WISDM sensor data from raw text files.

    Parameters:
        data_dir (str): Root path to WISDM dataset.
        file_limit (int, optional): Max number of files to load.

    Returns:
        pd.DataFrame: Concatenated sensor data.
    """
    patterns = [
        os.path.join(data_dir, 'raw', 'phone', 'accel', '*.txt'),
        os.path.join(data_dir, 'raw', 'phone', 'gyro', '*.txt'),
        os.path.join(data_dir, 'raw', 'watch', 'accel', '*.txt'),
        os.path.join(data_dir, 'raw', 'watch', 'gyro', '*.txt')
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if file_limit:
        files = files[:file_limit]

    frames = []
    for fpath in files:
        print(f"Loading {os.path.basename(fpath)}")
        try:
            df = pd.read_csv(
                fpath, header=None,
                names=['subject_id','activity','timestamp','x','y','z'],
                dtype={'subject_id':str,'activity':str}
            )
            # strip trailing semicolon in z
            df['z'] = df['z'].astype(str).str.rstrip(';').astype(float)
            
            # Add metadata from filename
            device = 'phone' if '/phone/' in fpath else 'watch'
            sensor = 'accel' if '/accel/' in fpath else 'gyro'
            df['device'] = device
            df['sensor_type'] = sensor
            
            frames.append(df)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def segment_data(df, fs=20, window_duration=5, overlap=0.5):
    """
    Split time series into windows per subject/activity.
    
    Parameters:
        df (pd.DataFrame): Raw sensor data
        fs (int): Sampling frequency in Hz
        window_duration (int): Window size in seconds
        overlap (float): Overlap between consecutive windows (0-1)
        
    Returns:
        list: List of DataFrame segments
    """
    wsize = int(window_duration * fs)
    step = int(wsize * (1 - overlap))
    segments = []
    
    # Group by subject, activity, device, and sensor type
    grouped = df.sort_values('timestamp').groupby(['subject_id', 'activity', 'device', 'sensor_type'])
    
    for i, ((subject, activity, device, sensor), group) in enumerate(grouped):
        n = len(group)
        
        if n < wsize:
            print(f"Group {i} (subject={subject}, activity={activity}) has only {n} samples, skipping")
            continue
        
        for start in range(0, n - wsize + 1, step):
            win = group.iloc[start:start+wsize].reset_index(drop=True)
            segments.append(win)
            
        if i % 10 == 0:
            print(f"Processed {i}/{len(grouped)} groups")
    
    print(f"Created {len(segments)} segments from {len(grouped)} groups")
    return segments


def compute_time_features(segment):
    """
    Compute time domain features for a single segment.
    
    Parameters:
        segment (pd.DataFrame): A single segment of sensor data
        
    Returns:
        dict: Dictionary of computed features
    """
    features = {}
    axes = ['x', 'y', 'z']
    
    # Get metadata
    features['subject_id'] = segment['subject_id'].iloc[0]
    features['activity'] = segment['activity'].iloc[0]
    features['device'] = segment['device'].iloc[0]
    features['sensor_type'] = segment['sensor_type'].iloc[0]
    
    for axis in axes:
        # Get signal values
        signal = segment[axis].values
        
        # Basic statistics
        features[f'{axis}_mean'] = np.mean(signal)
        features[f'{axis}_std'] = np.std(signal)
        features[f'{axis}_var'] = np.var(signal)
        features[f'{axis}_min'] = np.min(signal)
        features[f'{axis}_max'] = np.max(signal)
        features[f'{axis}_range'] = features[f'{axis}_max'] - features[f'{axis}_min']
        features[f'{axis}_median'] = np.median(signal)
        
        # Interquartile Range (IQR)
        q75, q25 = np.percentile(signal, [75, 25])
        features[f'{axis}_iqr'] = q75 - q25
        
        # Root Mean Square (RMS)
        features[f'{axis}_rms'] = np.sqrt(np.mean(np.square(signal)))
        
        # Zero-Crossing Rate
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        features[f'{axis}_zero_crossing_rate'] = len(zero_crossings) / len(signal)
        
        # Skewness and Kurtosis
        features[f'{axis}_skewness'] = stats.skew(signal)
        features[f'{axis}_kurtosis'] = stats.kurtosis(signal)
        
        # Energy
        features[f'{axis}_energy'] = np.sum(np.square(signal))
        
        # Peak detection
        peaks, _ = find_peaks(signal)
        features[f'{axis}_peak_count'] = len(peaks)
        
        if len(peaks) > 0:
            features[f'{axis}_peak_mean_amplitude'] = np.mean(signal[peaks])
            features[f'{axis}_peak_max_amplitude'] = np.max(signal[peaks])
        else:
            features[f'{axis}_peak_mean_amplitude'] = 0
            features[f'{axis}_peak_max_amplitude'] = 0
        
        # Autocorrelation (for lag=1)
        if len(signal) > 1:
            features[f'{axis}_autocorr'] = np.corrcoef(signal[:-1], signal[1:])[0, 1]
        else:
            features[f'{axis}_autocorr'] = 0
    
    # Calculate Signal Magnitude Area (SMA)
    x_signal = segment['x'].values
    y_signal = segment['y'].values
    z_signal = segment['z'].values
    sma = np.mean(np.abs(x_signal) + np.abs(y_signal) + np.abs(z_signal))
    features['sma'] = sma
    
    return features


def extract_features(segments):
    """
    Extract time domain features from all segments
    
    Parameters:
        segments (list): List of segment dataframes
        
    Returns:
        tuple: (X, y) feature matrix and activity labels
    """
    feat_dicts, labels = [], []
    for i, seg in enumerate(segments):
        try:
            features = compute_time_features(seg)
            feat_dicts.append(features)
            labels.append(seg['activity'].iloc[0])
            
            if (i+1) % 100 == 0:
                print(f"Processed {i+1}/{len(segments)} segments")
        except Exception as e:
            print(f"Error processing segment {i}: {e}")
    
    X = pd.DataFrame(feat_dicts)
    
    # Separate label column from features
    if 'activity' in X.columns:
        y = X['activity']
        meta = pd.DataFrame(feat_dicts)[['subject_id', 'activity', 'device', 'sensor_type']]
        meta.to_csv(os.path.join(output_dir, 'time_metadata.csv'), index=False)
        X = X.drop(columns=['activity', 'subject_id', 'device', 'sensor_type'])
    else:
        y = pd.Series(labels, name='activity')
    
    return X, y


def select_features(X, y, threshold='median'):
    """
    Select important features using Random Forest feature importance
    
    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target labels
        threshold (str): Feature importance threshold
        
    Returns:
        tuple: (X_selected, selected_features)
    """
    print("Selecting important features...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X.fillna(0), y)
    sel = SelectFromModel(rf, threshold=threshold, prefit=True)
    X_sel = sel.transform(X.fillna(0))
    cols = X.columns[sel.get_support()]
    return pd.DataFrame(X_sel, columns=cols), cols


if __name__ == '__main__':
    # This will run when you execute this script directly
    print("Extracting time domain features for WISDM dataset")
    
    cwd = os.path.dirname(__file__)
    data_root = os.path.abspath(os.path.join(cwd, '..', 'wisdm-dataset'))
    output_dir = os.path.abspath(os.path.join(cwd, '..', 'features'))
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    raw_df = load_raw_data(data_root, file_limit=None)
    print(f"Loaded raw data: {raw_df.shape}")
    
    # Segment data
    segs = segment_data(raw_df)
    
    # Extract time domain features
    X_all, y = extract_features(segs)
    print(f"Extracted {X_all.shape[1]} time domain features")
    
    # Save all features
    all_features_path = os.path.join(output_dir, 'time_all_features.csv')
    X_all.to_csv(all_features_path, index=False)
    print(f"Saved all features to {all_features_path}")
    
    # Select and save important features
    X_sel, feats_sel = select_features(X_all, y)
    print(f"Selected {len(feats_sel)} features")
    print("Top selected features:", list(feats_sel)[:20])
    
    # Save selected features
    selected_features_path = os.path.join(output_dir, 'time_selected_features.csv')
    X_sel.to_csv(selected_features_path, index=False)
    print(f"Saved selected features to {selected_features_path}")
    
    # Save labels
    labels_path = os.path.join(output_dir, 'activity_labels.csv')
    y.to_csv(labels_path, index=False)
    print(f"Saved activity labels to {labels_path}")