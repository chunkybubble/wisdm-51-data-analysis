import os
import glob
import pandas as pd
import numpy as np
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
        df = pd.read_csv(
            fpath, header=None,
            names=['subject_id','activity','timestamp','x','y','z'],
            dtype={'subject_id':str,'activity':str}
        )
        # strip trailing semicolon in z
        df['z'] = df['z'].astype(str).str.rstrip(';').astype(float)
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def segment_data(df, fs=20, window_duration=10, overlap=0.0):
    """
    Split time series into windows per subject/activity.
    """
    wsize = int(window_duration * fs)
    step = int(wsize * (1 - overlap))
    segments = []
    grouped = df.sort_values('timestamp').groupby(['subject_id','activity'])
    for (_,act_group), group in grouped:
        n = len(group)
        for start in range(0, n - wsize + 1, step):
            win = group.iloc[start:start+wsize].reset_index(drop=True)
            segments.append(win)
    return segments


def compute_frequency_features(seg, fs=20, n_coeffs=10):
    """
    Compute per-axis frequency-domain features: centroid, entropy,
    energy, peak freq, variance, flatness, bandwidth, FFT & PSD coefficients.
    """
    feats = {}
    axes = ['x','y','z']
    N = len(seg)
    n_fft = N  # no zero-padding

    for axis in axes:
        data = seg[axis].astype(float).values
        data = data - data.mean()

        # FFT and power spectrum
        fft_vals = np.fft.rfft(data, n=n_fft)
        mags = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(n_fft, 1/fs)
        psd = mags**2
        total_mag = mags.sum() + 1e-12
        total_psd = psd.sum() + 1e-12

        # Spectral Centroid
        centroid = (freqs * mags).sum() / total_mag
        feats[f'{axis}_spectral_centroid'] = centroid

        # Spectral Entropy
        p_norm = psd / total_psd
        entropy = - (p_norm * np.log2(p_norm + 1e-12)).sum()
        feats[f'{axis}_spectral_entropy'] = entropy

        # Spectral Energy
        feats[f'{axis}_spectral_energy'] = total_psd

        # Peak Frequency
        peak_idx = np.argmax(mags)
        peak_freq = freqs[peak_idx]
        feats[f'{axis}_dominant_frequency'] = peak_freq
        feats[f'{axis}_peak_frequency'] = peak_freq

        # Frequency Variance
        var = ((freqs - centroid)**2 * mags).sum() / total_mag
        feats[f'{axis}_frequency_variance'] = var

        # Spectral Flatness
        geo = np.exp(np.log(psd + 1e-12).mean())
        arith = p_norm.mean() * total_psd  # p_norm.mean()*total_psd == arith? simpler: psd.mean(); use psd.mean()
        arith = psd.mean()
        feats[f'{axis}_spectral_flatness'] = geo / (arith + 1e-12)

        # Bandwidth (5%-95% cumulative PSD)
        cum = np.cumsum(psd)
        cum_norm = cum / total_psd
        if total_psd > 1e-8:
            low = freqs[np.searchsorted(cum_norm, 0.05)]
            high = freqs[np.searchsorted(cum_norm, 0.95)]
            bandwidth = high - low
        else:
            bandwidth = 0.0
        feats[f'{axis}_bandwidth'] = bandwidth

        # First n_coeffs FFT coefficients
        for i in range(n_coeffs):
            feats[f'{axis}_fft_coeff_{i}'] = np.real(fft_vals[i])

        # First n_coeffs PSD values
        for i in range(n_coeffs):
            feats[f'{axis}_psd_{i}'] = psd[i] if i < len(psd) else 0.0

    return feats


def extract_features(segments, fs=20):
    feat_dicts, labels = [], []
    for seg in segments:
        feat_dicts.append(compute_frequency_features(seg, fs))
        labels.append(seg['activity'].iat[0])
    X = pd.DataFrame(feat_dicts)
    y = pd.Series(labels, name='activity')
    return X, y


def select_features(X, y, threshold='median'):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X.fillna(0), y)
    sel = SelectFromModel(rf, threshold=threshold, prefit=True)
    X_sel = sel.transform(X.fillna(0))
    cols = X.columns[sel.get_support()]
    return pd.DataFrame(X_sel, columns=cols), cols


if __name__ == '__main__':
    cwd = os.path.dirname(__file__)
    data_root = os.path.abspath(os.path.join(cwd, '..', 'wisdm-dataset'))
    raw_df = load_raw_data(data_root)
    segs = segment_data(raw_df)
    X_all, y = extract_features(segs)
    X_sel, feats_sel = select_features(X_all, y)
    print("Selected features:", list(feats_sel))
    X_all.to_csv('frequency_all_features.csv', index=False)
    X_sel.to_csv('frequency_selected_features.csv', index=False)
