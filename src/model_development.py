import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import time


def load_data(features_dir):
    """
    Load the extracted features and labels.
    
    Parameters:
        features_dir (str): Directory containing feature CSV files
        
    Returns:
        tuple: (X_time, X_freq, y) feature matrices and labels
    """
    # Load time domain features
    time_features_path = os.path.join(features_dir, 'time_selected_features.csv')
    X_time = pd.read_csv(time_features_path)
    print(f"Loaded time domain features: {X_time.shape}")
    
    # Load frequency domain features
    freq_features_path = os.path.join(features_dir, 'frequency_selected_features.csv')
    X_freq = pd.read_csv(freq_features_path)
    print(f"Loaded frequency domain features: {X_freq.shape}")
    
    # Load activity labels
    labels_path = os.path.join(features_dir, 'activity_labels.csv')
    y = pd.read_csv(labels_path).squeeze()
    print(f"Loaded activity labels: {y.shape}")
    
    return X_time, X_freq, y


def prepare_feature_sets(X_time, X_freq, y):
    """
    Prepare different feature combinations for model evaluation.
    
    Parameters:
        X_time (pd.DataFrame): Time domain features
        X_freq (pd.DataFrame): Frequency domain features
        y (pd.Series): Activity labels
        
    Returns:
        dict: Dictionary containing different feature sets and their names
    """
    feature_sets = {
        'time_features': (X_time, y, 'Time Domain Features'),
        'freq_features': (X_freq, y, 'Frequency Domain Features')
    }
    
    # Combine time and frequency features if both are available
    if X_time is not None and X_freq is not None:
        # Get common indices to ensure proper merging
        X_combined = pd.concat([X_time, X_freq], axis=1)
        feature_sets['combined_features'] = (X_combined, y, 'Combined Features')
    
    return feature_sets


def train_evaluate_models(feature_sets, test_size=0.2, random_state=42):
    """
    Train and evaluate different ML models on the feature sets.
    
    Parameters:
        feature_sets (dict): Dictionary of feature sets
        test_size (float): Test split ratio
        random_state (int): Random state for reproducibility
        
    Returns:
        dict: Dictionary containing results for each model and feature set
    """
    results = {}
    
    # Define models to evaluate
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state)
    }
    
    # Model specific parameter grids for GridSearchCV
    param_grids = {
        'Decision Tree': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30]
        }
    }
    
    # Iterate through feature sets
    for feature_name, (X, y, feature_desc) in feature_sets.items():
        print(f"\n==== Evaluating models using {feature_desc} ====")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        feature_results = {}
        
        # Evaluate each model
        for model_name, model in models.items():
            start_time = time.time()
            print(f"\nTraining {model_name}...")
            
            # Create a pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            # Set up cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            
            # GridSearch for hyperparameter tuning
            param_grid = {'classifier__' + key: value for key, value in param_grids[model_name].items()}
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Store results
            training_time = time.time() - start_time
            feature_results[model_name] = {
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix': conf_matrix,
                'best_params': grid_search.best_params_,
                'training_time': training_time,
                'model': best_model
            }
            
            print(f"{model_name} - Accuracy: {accuracy:.4f}, Training Time: {training_time:.2f}s")
            print(f"Best parameters: {grid_search.best_params_}")
        
        results[feature_name] = feature_results
    
    return results


def visualize_results(results):
    """
    Visualize the performance of different models.
    
    Parameters:
        results (dict): Results from model evaluation
    """
    # Plot accuracy comparison
    plt.figure(figsize=(12, 8))
    
    feature_sets = list(results.keys())
    model_names = list(results[feature_sets[0]].keys())
    
    x = np.arange(len(model_names))
    width = 0.2
    offsets = np.linspace(-(len(feature_sets) - 1) * width / 2, (len(feature_sets) - 1) * width / 2, len(feature_sets))
    
    for i, feature_name in enumerate(feature_sets):
        accuracies = [results[feature_name][model]['accuracy'] for model in model_names]
        plt.bar(x + offsets[i], accuracies, width, label=feature_name)
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('model_performance_comparison.png')
    plt.close()
    
    # Plot confusion matrices for the best model
    best_accuracy = 0
    best_model_info = None
    
    for feature_name in feature_sets:
        for model_name in model_names:
            accuracy = results[feature_name][model_name]['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_info = (feature_name, model_name)
    
    if best_model_info:
        feature_name, model_name = best_model_info
        cm = results[feature_name][model_name]['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name} with {feature_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name}_{feature_name}.png')
        plt.close()


def save_results(results, output_dir):
    """
    Save model evaluation results to files.
    
    Parameters:
        results (dict): Results from model evaluation
        output_dir (str): Directory to save results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save accuracy results to CSV
    accuracy_data = []
    for feature_name in results:
        for model_name in results[feature_name]:
            accuracy_data.append({
                'Feature Set': feature_name,
                'Model': model_name,
                'Accuracy': results[feature_name][model_name]['accuracy'],
                'Training Time': results[feature_name][model_name]['training_time'],
                'Best Parameters': results[feature_name][model_name]['best_params']
            })
    
    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_df.to_csv(os.path.join(output_dir, 'model_accuracies.csv'), index=False)
    
    # Find and save best model
    best_accuracy = 0
    best_model_info = None
    
    for feature_name in results:
        for model_name in results[feature_name]:
            accuracy = results[feature_name][model_name]['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_info = (feature_name, model_name, results[feature_name][model_name]['model'])
    
    if best_model_info:
        feature_name, model_name, best_model = best_model_info
        with open(os.path.join(output_dir, 'best_model_info.txt'), 'w') as f:
            f.write(f"Best Model: {model_name}\n")
            f.write(f"Feature Set: {feature_name}\n")
            f.write(f"Accuracy: {best_accuracy:.4f}\n")
            f.write(f"Parameters: {results[feature_name][model_name]['best_params']}\n")
        
        # Save detailed classification report
        report_df = pd.DataFrame(results[feature_name][model_name]['report'])
        report_df.to_csv(os.path.join(output_dir, 'best_model_classification_report.csv'))
        
        # Optional: Save the model itself
        # import joblib
        # joblib.dump(best_model, os.path.join(output_dir, 'best_model.joblib'))


if __name__ == '__main__':
    # Path configurations
    cwd = os.path.dirname(__file__)
    features_dir = os.path.abspath(os.path.join(cwd, '..', 'features'))
    output_dir = os.path.abspath(os.path.join(cwd, '..', 'model_evaluation'))
    
    print("Model Development for Human Activity Recognition")
    print("=" * 50)
    
    # Load data
    X_time, X_freq, y = load_data(features_dir)
    
    # Prepare feature sets
    feature_sets = prepare_feature_sets(X_time, X_freq, y)
    
    # Train and evaluate models
    results = train_evaluate_models(feature_sets)
    
    # Visualize results
    visualize_results(results)
    
    # Save results
    save_results(results, output_dir)
    
    print("\nModel evaluation completed.")
    print(f"Results saved to {output_dir}")