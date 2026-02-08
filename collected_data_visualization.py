import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Constants
INPUT_DIR = 'collected_data'
OUTPUT_DIR = 'collected_data_visuals'
SAMPLE_RATE = 50  # Hz
SLICE_SEC = 10    # Plot first 10s of a sample

def setup_env():
    """Ensure output directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load all CSV files from the input directory and combine them."""
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    return pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

def preprocess(df):
    """Calculate magnitude and relative time for plotting."""
    df['mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['rel_sec'] = (df['sample_ts_ms'] - df.groupby('window_id')['sample_ts_ms'].transform('min')) / 1000
    # Global time for inter-window interval analysis
    df['dt_ms'] = df['sample_ts_ms'].diff()
    return df

def plot_timeseries(df, label, col, title, filename):
    """Generate time-series plot for a specific activity slice."""
    subset = df[df['label'] == label].head(SLICE_SEC * SAMPLE_RATE)
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(subset)) / SAMPLE_RATE, subset[col], color='#2196F3' if col=='ax' else '#FF5722')
    plt.title(f"{title} ({label} - First {SLICE_SEC}s)", fontsize=12, fontweight='bold')
    plt.xlabel("Time (s)"); plt.ylabel(col); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()

def plot_histogram(df):
    """Plot sampling interval distribution."""
    intervals = df['dt_ms'].dropna()
    intervals = intervals[intervals < (1000/SAMPLE_RATE * 3)] # Remove outlier gaps
    plt.figure(figsize=(10, 5))
    plt.hist(intervals, bins=50, color='#4CAF50', edgecolor='white')
    plt.axvline(1000/SAMPLE_RATE, color='red', linestyle='--', label=f'Target ({1000/SAMPLE_RATE}ms)')
    plt.title("Sampling Interval Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("Interval (ms)"); plt.ylabel("Frequency"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'sampling_intervals.png'), dpi=150)
    plt.close()

def plot_activity_dist(df):
    """Show distribution of windows per activity."""
    counts = df.groupby('label')['window_id'].nunique()
    plt.figure(figsize=(10, 5))
    counts.plot(kind='bar', color='#673AB7', edgecolor='white')
    plt.title("Activity Distribution (Unique Windows)", fontsize=12, fontweight='bold')
    plt.xlabel("Activity"); plt.ylabel("Count"); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'activity_distribution.png'), dpi=150)
    plt.close()

def main():
    setup_env()
    print("Loading data..."); df = load_data()
    print("Preprocessing..."); df = preprocess(df)
    
    # Generate plots
    sample_label = df['label'].iloc[0]
    print(f"Plotting time series for {sample_label}...")
    plot_timeseries(df, sample_label, 'ax', 'X-Acceleration', 'x_accel_vs_time.png')
    plot_timeseries(df, sample_label, 'mag', 'Acceleration Magnitude', 'magnitude_vs_time.png')
    
    print("Plotting distribution and intervals...")
    plot_histogram(df)
    plot_activity_dist(df)
    
    print(f"\nSuccess! Visuals saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()
