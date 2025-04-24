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