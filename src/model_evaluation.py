import os
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    GroupKFold,
    LeaveOneGroupOut,
    cross_validate,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.pipeline import Pipeline

# Define scoring metrics (macro-average)
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro', zero_division=0),
    'recall': make_scorer(recall_score, average='macro', zero_division=0),
    'f1': make_scorer(f1_score, average='macro', zero_division=0)
}

# Models and hyperparameter grids
models = {
    'Decision Tree': DecisionTreeClassifier,
    'KNN': KNeighborsClassifier,
    'Random Forest': RandomForestClassifier
}

param_grids = {
    'Decision Tree': {'max_depth': [None, 20, 30], 'min_samples_split': [2, 10]},
    'KNN': {'n_neighbors': [3, 5, 9], 'weights': ['uniform', 'distance']},
    'Random Forest': {'n_estimators': [50, 100], 'max_depth': [None, 20]}
}


def load_data(feature_path, label_path, metadata_path=None):
    """
    Load features (X), labels (y), and optional grouping metadata.
    Returns X, y, subject_groups, episode_groups.
    """
    X = pd.read_csv(feature_path)
    y = pd.read_csv(label_path).squeeze()
    n = min(len(X), len(y))
    X = X.iloc[:n]
    y = y.iloc[:n]
    subject_groups, episode_groups = None, None
    if metadata_path:
        meta = pd.read_csv(metadata_path).iloc[:n]
        if 'subject_id' in meta.columns:
            subject_groups = meta['subject_id']
        if set(['activity', 'device', 'sensor_type']).issubset(meta.columns):
            episode_groups = (
                meta['subject_id'].astype(str) + '_' +
                meta['activity'] + '_' +
                meta['device'] + '_' +
                meta['sensor_type']
            )
    return X, y, subject_groups, episode_groups


def evaluate(X, y, cv, cv_name):
    """
    Perform grid search (inner CV) then cross-validate best estimators.
    - X, y: data
    - cv: cross-validation splitter or generator
    - cv_name: label for printing
    Prints mean ± std for each metric and model.
    """
    print(f"\n--- {cv_name} ---")
    for name, Model in models.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', Model())])
        grid = GridSearchCV(
            pipe,
            param_grid={f'clf__{k}': v for k, v in param_grids[name].items()},
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='accuracy', n_jobs=-1
        )
        grid.fit(X, y)
        best = grid.best_estimator_
        cv_res = cross_validate(best, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        print(f"\n{name}:")
        for metric in scoring.keys():
            scores = cv_res[f'test_{metric}']
            print(f"  {metric}: {scores.mean():.3f} ± {scores.std():.3f}")


def main():
    cwd = os.path.dirname(__file__)
    feat_dir = os.path.abspath(os.path.join(cwd, '..', 'features'))

    # File paths
    time_feat = os.path.join(feat_dir, 'time_selected_features.csv')
    freq_feat = os.path.join(feat_dir, 'frequency_selected_features.csv')
    labels = os.path.join(feat_dir, 'activity_labels.csv')
    time_meta = os.path.join(feat_dir, 'time_all_features.csv')
    freq_meta = os.path.join(feat_dir, 'frequency_all_features.csv')

    # Time-domain evaluation
    X_time, y_time, sub_time, epi_time = load_data(time_feat, labels, time_meta)
    evaluate(X_time, y_time, StratifiedKFold(n_splits=10, shuffle=True, random_state=42), 'Time 10-Fold CV')
    if sub_time is not None:
        evaluate(X_time, y_time, GroupKFold(n_splits=sub_time.nunique()).split(X_time, y_time, sub_time), 'Time LOSO CV')
    if epi_time is not None:
        evaluate(X_time, y_time, LeaveOneGroupOut().split(X_time, y_time, epi_time), 'Time LOEO CV')

    # Frequency-domain evaluation
    X_freq, y_freq, sub_freq, epi_freq = load_data(freq_feat, labels, freq_meta)
    evaluate(X_freq, y_freq, StratifiedKFold(n_splits=10, shuffle=True, random_state=42), 'Freq 10-Fold CV')
    if sub_freq is not None:
        evaluate(X_freq, y_freq, GroupKFold(n_splits=sub_freq.nunique()).split(X_freq, y_freq, sub_freq), 'Freq LOSO CV')
    if epi_freq is not None:
        evaluate(X_freq, y_freq, LeaveOneGroupOut().split(X_freq, y_freq, epi_freq), 'Freq LOEO CV')

if __name__ == '__main__':
    main()
