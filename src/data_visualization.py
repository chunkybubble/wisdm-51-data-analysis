import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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