import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re

# Set the style for our plots
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)

def read_arff_file(file_path):
    """
    Read ARFF file format from the WISDM dataset
    
    Parameters:
    file_path (str): Path to the ARFF file
    
    Returns:
    pandas.DataFrame: Processed data from the file
    """
    try:
        # Read ARFF file as text first
        with open(file_path, 'r') as f:
            content = f.readlines()
        
        # Find the data section
        data_start = None
        headers = []
        
        for i, line in enumerate(content):
            line = line.strip()
            
            # Extract attribute names for headers
            if line.lower().startswith('@attribute'):
                attr_match = re.match(r'@attribute\s+(\S+)', line, re.IGNORECASE)
                if attr_match:
                    headers.append(attr_match.group(1))
            
            # Find the start of data section
            if line.lower() == '@data':
                data_start = i + 1
                break
        
        if data_start is None:
            print(f"No @data section found in {file_path}")
            return None
        
        # Extract actual data
        data_lines = [line.strip() for line in content[data_start:] if line.strip() and not line.startswith('%')]
        
        # Convert to DataFrame
        data = [line.split(',') for line in data_lines]
        df = pd.DataFrame(data, columns=headers)
        
        # Convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
        
        file_name = os.path.basename(file_path)
        match = re.match(r'data_(\d+)_(\w+)_(\w+)\.arff', file_name)
        
        if match:
            subject_id, sensor_type, device = match.groups()
            df['subject_id'] = subject_id
            df['sensor_type'] = sensor_type
            df['device'] = device
        
        return df
    except Exception as e:
        print(f"Error reading ARFF file {file_path}: {e}")
        return None

def load_wisdm_data(data_dir, file_limit=None):
    """
    Load and process WISDM dataset files.
    
    Parameters:
    data_dir (str): Directory containing the WISDM dataset files
    file_limit (int, optional): Maximum number of files to load (for testing)
    
    Returns:
    pandas.DataFrame: Processed dataset
    """
    print(f"Looking for data in: {data_dir}")
    
    # Check if the directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    # Look for ARFF files
    arff_files = []
    arff_phone_accel = glob.glob(os.path.join(data_dir, "arff_files/phone/accel/*.arff"))
    arff_phone_gyro = glob.glob(os.path.join(data_dir, "arff_files/phone/gyro/*.arff"))
    arff_watch_accel = glob.glob(os.path.join(data_dir, "arff_files/watch/accel/*.arff"))
    arff_watch_gyro = glob.glob(os.path.join(data_dir, "arff_files/watch/gyro/*.arff"))
    
    arff_files = arff_phone_accel + arff_phone_gyro + arff_watch_accel + arff_watch_gyro
    print(f"Found {len(arff_files)} ARFF files (.arff)")
    
    # Limit files for testing if requested
    if file_limit:
        arff_files = arff_files[:file_limit]
        print(f"Loading {file_limit} ARFF files for testing")
    
    # Load ARFF data
    arff_data = []
    if arff_files:
        print("Loading ARFF files (this might take a while)...")
        for i, file in enumerate(arff_files):
            print(f"Processing ARFF file {i+1}/{len(arff_files)}: {os.path.basename(file)}")
            df = read_arff_file(file)
            if df is not None:
                arff_data.append(df)
        
        if arff_data:
            print(f"Successfully loaded {len(arff_data)} ARFF files")
            arff_df = pd.concat(arff_data, ignore_index=True)
            print(f"Combined ARFF data shape: {arff_df.shape}")
            return arff_df
        else:
            print("Failed to load any ARFF files")
            return None
    else:
        print("No ARFF files found")
        return None

def visualize_arff_data(df, output_dir='visualizations/arff', show_plots=False):
    """
    Create visualizations for ARFF data format
    
    Parameters:
    df (pandas.DataFrame): ARFF data
    output_dir (str): Directory to save visualizations
    show_plots (bool): Whether to display plots interactively
    """
    if df is None or df.empty:
        print("No ARFF data to visualize")
        return
    
    print("\nARFF Data Summary:")
    print(df.info())
    print("\nARFF Data Sample:")
    print(df.head())
    
    # Create a folder for saving visualizations
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric columns: {len(numeric_cols)} found")
    print(f"Categorical columns: {len(categorical_cols)} found")
    
    # 1. Activity/Class distribution
    activity_col = None
    for col in categorical_cols:
        if 'activity' in col.lower():
            activity_col = col
            break
    
    if activity_col:
        plt.figure(figsize=(12, 8))
        activity_counts = df[activity_col].value_counts()
        sns.barplot(x=activity_counts.index, y=activity_counts.values)
        plt.title(f'Distribution of Activities')
        plt.xlabel('Activity')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/activity_distribution.png')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 2. Histograms for important features 
    # Looking for acceleration data and features
    accel_features = [col for col in numeric_cols if any(axis in col.lower() for axis in ['x', 'y', 'z'])]
    for col in accel_features[:5]:  # First 5 acceleration features
        plt.figure(figsize=(10, 6))
        plt.hist(df[col].dropna(), bins=30, alpha=0.7)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        safe_col = col.replace('"', '')
        plt.savefig(f'{output_dir}/{safe_col}_histogram.png')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 3. Scatter plots - comparing X/Y/Z axes
    # TODO: fix scatterplot
    x_cols = [col for col in numeric_cols if col.lower().startswith('x')]
    y_cols = [col for col in numeric_cols if col.lower().startswith('y')]
    z_cols = [col for col in numeric_cols if col.lower().startswith('z')]
    
    if x_cols and y_cols and z_cols:
        # X vs Y (using first of each)
        plt.figure(figsize=(10, 10))
        plt.scatter(df[x_cols[0]], df[y_cols[0]], alpha=0.5)
        plt.title(f'{x_cols[0]} vs {y_cols[0]}')
        plt.xlabel(x_cols[0])
        plt.ylabel(y_cols[0])
        plt.tight_layout()
        plt.savefig(f'{output_dir}/x_vs_y.png')
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # X vs Z
        plt.figure(figsize=(10, 10))
        plt.scatter(df[x_cols[0]], df[z_cols[0]], alpha=0.5)
        plt.title(f'{x_cols[0]} vs {z_cols[0]}')
        plt.xlabel(x_cols[0])
        plt.ylabel(z_cols[0])
        plt.tight_layout()
        plt.savefig(f'{output_dir}/x_vs_z.png')
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Y vs Z
        plt.figure(figsize=(10, 10))
        plt.scatter(df[y_cols[0]], df[z_cols[0]], alpha=0.5)
        plt.title(f'{y_cols[0]} vs {z_cols[0]}')
        plt.xlabel(y_cols[0])
        plt.ylabel(z_cols[0])
        plt.tight_layout()
        plt.savefig(f'{output_dir}/y_vs_z.png')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 4. Pair plot with activity classes
    if activity_col and len(accel_features) >= 3:
        # Select a subset of columns for the pair plot
        selected_cols = accel_features[:3]  # First 3 acceleration features
        
        # Sample data to avoid overwhelming the plot
        sample_df = df[selected_cols + [activity_col]].sample(min(3000, len(df)))
        
        # Create pair plot
        plt.figure(figsize=(15, 15))
        g = sns.pairplot(sample_df, hue=activity_col, plot_kws={'alpha': 0.5}, diag_kind='kde')
        g.fig.suptitle('Pairwise Relationships between Features', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pair_plot.png')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 5. Compare device types
    if 'device' in df.columns and accel_features:
        plt.figure(figsize=(14, 8))
        for device in df['device'].unique():
            device_data = df[df['device'] == device]
            sns.kdeplot(device_data[accel_features[0]], label=f'{device}')
        plt.title(f'{accel_features[0]} Distribution by Device')
        plt.xlabel(accel_features[0])
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/device_comparison.png')
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    # 6. Compare sensor types
    if 'sensor_type' in df.columns and accel_features:
        plt.figure(figsize=(14, 8))
        for sensor in df['sensor_type'].unique():
            sensor_data = df[df['sensor_type'] == sensor]
            sns.kdeplot(sensor_data[accel_features[0]], label=f'{sensor}')
        plt.title(f'{accel_features[0]} Distribution by Sensor Type')
        plt.xlabel(accel_features[0])
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sensor_comparison.png')
        if show_plots:
            plt.show()
        else:
            plt.close()

# Main execution
if __name__ == "__main__":
    # Specify the path to your data
    data_dir = "wisdm-dataset"
    data_dir = os.path.expanduser(data_dir)  # Expand the ~ to the home directory
    
    print("Starting WISDM dataset analysis...")
    
    # Load all ARFF data or limit for testing
    df = load_wisdm_data(data_dir, file_limit=20)  # Increase file_limit to process more files
    
    if df is not None:
        # Create visualizations
        visualize_arff_data(df, show_plots=True)  # Set show_plots=True to display plots interactively
        print("Data visualization complete. Check the 'visualizations/arff' folder for results.")
    else:
        print("Failed to load any data. Please check the data directory and file format.")