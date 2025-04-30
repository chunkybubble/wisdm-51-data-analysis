import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def load_raw_data(data_dir, file_limit=None):
    """
    Load raw WISDM sensor data from text files, adding metadata.
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
            # cleanup
            df['z'] = df['z'].astype(str).str.rstrip(';').astype(float)
            path = fpath.replace('\\','/')
            df['device'] = 'phone' if '/phone/' in path else 'watch'
            df['sensor_type'] = 'accel' if '/accel/' in path else 'gyro'
            frames.append(df)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def segment_data(df, fs=20, window_duration=5, overlap=0.5):
    """
    Create overlapping windows per subject/activity/device/sensor.
    """
    wsize = int(window_duration * fs)
    step = int(wsize * (1-overlap))
    segments = []
    groups = df.sort_values('timestamp').groupby(
        ['subject_id','activity','device','sensor_type']
    )
    for idx, (_, grp) in enumerate(groups):
        if len(grp) < wsize:
            continue
        for start in range(0, len(grp)-wsize+1, step):
            win = grp.iloc[start:start+wsize].reset_index(drop=True)
            segments.append(win)
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(groups)} groups")
    print(f"Created {len(segments)} segments from {len(groups)} groups")
    return segments


def compute_frequency_features(segment, fs=20, n_coeffs=10):
    """
    Compute frequency-domain features for one window (segment).
    """
    feats = {
        'subject_id': segment['subject_id'].iat[0],
        'activity': segment['activity'].iat[0],
        'device': segment['device'].iat[0],
        'sensor_type': segment['sensor_type'].iat[0]
    }
    N = len(segment)
    n_fft = N
    for axis in ['x','y','z']:
        data = segment[axis].astype(float).values
        data_centered = data - data.mean()
        fft_vals = np.fft.rfft(data_centered, n=n_fft)
        mags = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(n_fft, 1/fs)
        psd = mags**2
        total_mag = mags.sum() + 1e-12
        total_psd = psd.sum() + 1e-12

        # spectral centroid
        centroid = (freqs * mags).sum() / total_mag
        feats[f'{axis}_spectral_centroid'] = centroid

        # spectral entropy
        p_norm = psd / total_psd
        feats[f'{axis}_spectral_entropy'] = - (p_norm * np.log2(p_norm + 1e-12)).sum()

        # spectral energy
        feats[f'{axis}_spectral_energy'] = total_psd

        # peak & dominant frequency
        peak_idx = np.argmax(mags)
        feats[f'{axis}_peak_frequency'] = freqs[peak_idx]
        feats[f'{axis}_dominant_frequency'] = freqs[peak_idx]

        # frequency variance
        feats[f'{axis}_frequency_variance'] = ((freqs-centroid)**2 * mags).sum() / total_mag

        # spectral flatness
        geo = np.exp(np.mean(np.log(psd + 1e-12)))
        arith = psd.mean()
        feats[f'{axis}_spectral_flatness'] = geo / (arith + 1e-12)

        # bandwidth (5-95%)
        cum = np.cumsum(psd) / total_psd
        if total_psd > 1e-8:
            low = freqs[np.searchsorted(cum,0.05)]
            high = freqs[np.searchsorted(cum,0.95)]
            feats[f'{axis}_bandwidth'] = high - low
        else:
            feats[f'{axis}_bandwidth'] = 0.0

        # FFT & PSD coeffs
        for i in range(n_coeffs):
            feats[f'{axis}_fft_coeff_{i}'] = np.real(fft_vals[i]) if i < len(fft_vals) else 0.0
            feats[f'{axis}_psd_{i}'] = psd[i] if i < len(psd) else 0.0
    return feats


def extract_features(segments):
    """
    Loop through windows, compute features, return (X, y).
    """
    all_feats = []
    for i, seg in enumerate(segments):
        try:
            all_feats.append(compute_frequency_features(seg))
            if (i+1) % 100 == 0:
                print(f"Processed {i+1}/{len(segments)} segments")
        except Exception as e:
            print(f"Error on segment {i}: {e}")
    X = pd.DataFrame(all_feats)
    if 'activity' in X:
        y = X['activity']
        meta = pd.DataFrame(all_feats)[['subject_id', 'activity', 'device', 'sensor_type']]
        meta.to_csv(os.path.join(out_dir, 'freq_metadata.csv'), index=False)
        X = X.drop(columns=['activity','subject_id','device','sensor_type'])
    else:
        y = pd.Series(name='activity')
    return X, y


def select_features(X, y, threshold='median'):
    """
    RF-based feature selection.
    """
    print("Selecting important frequency features...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X.fillna(0), y)
    sel = SelectFromModel(rf, threshold=threshold, prefit=True)
    X_sel = sel.transform(X.fillna(0))
    cols = X.columns[sel.get_support()]
    return pd.DataFrame(X_sel,columns=cols), cols


if __name__ == '__main__':
    print("Running frequency feature extraction...")
    base = os.path.dirname(__file__)
    data_root = os.path.abspath(os.path.join(base,'..','wisdm-dataset'))
    out_dir = os.path.abspath(os.path.join(base,'..','features'))
    os.makedirs(out_dir,exist_ok=True)

    raw = load_raw_data(data_root)
    print(f"Loaded raw: {raw.shape}")
    segs = segment_data(raw)
    X_all, y = extract_features(segs)
    print(f"Extracted features matrix {X_all.shape}")

    fp = os.path.join(out_dir,'frequency_all_features.csv')
    X_all.to_csv(fp,index=False)
    print(f"Saved all to {fp}")

    X_sel, feats = select_features(X_all,y)
    print(f"Selected {len(feats)} features")

    sp = os.path.join(out_dir,'frequency_selected_features.csv')
    X_sel.to_csv(sp,index=False)
    print(f"Saved selected to {sp}")

    # lp = os.path.join(out_dir,'activity_labels.csv')
    # y.to_csv(lp,index=False)
    # print(f"Saved labels to {lp}")
