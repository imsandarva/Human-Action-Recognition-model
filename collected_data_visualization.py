import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Constants
INPUT_DIR, OUTPUT_DIR = 'collected_data', 'collected_data_visuals'

def setup():
    """Ensure output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_data():
    """Load and combine all CSV files from collected_data."""
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    return pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

def add_features(df):
    """Calculate magnitude and intervals to mirror existing logic."""
    df['mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['dt_ms'] = df.groupby('window_id')['sample_ts_ms'].diff()
    return df

def save_plot(filename):
    """Save plot and close to free memory."""
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150); plt.close()

def plot_full_timeseries(df, col, title, file):
    """Plot full time series (all windows) for the specified column."""
    plt.figure(figsize=(15, 6)); plt.plot(df.index / 50, df[col], color='#2196F3', linewidth=0.5)
    plt.title(f"Full Collected Data: {title}"); plt.xlabel("Estimated Time (s)"); plt.ylabel(col); plt.grid(alpha=0.3)
    save_plot(file)

def plot_intervals(df):
    """Mirror sampling interval histogram logic."""
    intervals = df['dt_ms'].dropna()
    plt.figure(figsize=(10, 6)); plt.hist(intervals[intervals < 100], bins=50, color='#4CAF50')
    plt.title("Sampling Interval Histogram (All Collected Data)"); plt.xlabel("Interval (ms)"); plt.ylabel("Frequency")
    save_plot('sampling_interval_histogram.png')

def plot_labels(df):
    """Mirror activity count bar chart logic."""
    counts = df.groupby('label')['window_id'].nunique()
    plt.figure(figsize=(10, 6)); counts.plot(kind='bar', color='#673AB7')
    plt.title("Activity Count (Unique Windows)"); plt.xlabel("Activity"); plt.ylabel("Count"); plt.xticks(rotation=45)
    save_plot('activity_count_bar_chart.png')

def main():
    setup(); print("Loading full dataset..."); df = load_all_data()
    print("Mirroring preprocessing logic..."); df = add_features(df)
    
    print("Generating full-scale visuals..."); 
    plot_full_timeseries(df, 'ax', 'X-Acceleration vs Time', 'x_accel_vs_time.png')
    plot_full_timeseries(df, 'mag', 'Acceleration Magnitude vs Time', 'magnitude_vs_time.png')
    plot_intervals(df); plot_labels(df)
    print(f"Success! Full visuals saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()
