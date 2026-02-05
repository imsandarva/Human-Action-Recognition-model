import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d

# Constants
DATA_PATH = 'wisdm_dataset/WISDM_ar_v1.1_raw.txt'
OUTPUT_DIR = 'processed_visuals'
PROCESSED_DATA_PATH = 'processed_wisdm.npz'
SAMPLING_RATE = 50  # Hz
WINDOW_SIZE_SEC = 2
OVERLAP_PERCENT = 0.5
MAX_GAP_SEC = 0.5  # Max gap to interpolate across

COLUMN_NAMES = [
    'user', 'activity', 'timestamp', 
    'x-acceleration', 'y-acceleration', 'z-acceleration'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading and cleaning raw data...")
try:
    df = pd.read_csv(
        DATA_PATH, 
        header=None, 
        names=COLUMN_NAMES, 
        sep=',', 
        decimal='.', 
        on_bad_lines='skip'
    )
    # Remove trailing semicolon and convert to numeric
    df['z-acceleration'] = df['z-acceleration'].astype(str).str.replace(';', '')
    df['z-acceleration'] = pd.to_numeric(df['z-acceleration'], errors='coerce')
    
    # Remove malformed/duplicate rows
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['user', 'activity', 'timestamp'], inplace=True)
    
    # Sort by user and timestamp
    df.sort_values(by=['user', 'timestamp'], inplace=True)
    
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

def resample_segment(segment_df):
    """Resamples a continuous segment of sensor data to 50Hz."""
    if len(segment_df) < 2:
        return None
    
    t_raw = segment_df['timestamp'].values / 1e9 # Convert nanoseconds to seconds
    t_start = t_raw[0]
    t_end = t_raw[-1]
    
    # Duration in seconds
    duration = t_end - t_start
    if duration < WINDOW_SIZE_SEC:
        return None
    
    # Target time grid (50Hz = 0.02s steps)
    t_target = np.arange(t_start, t_end, 1.0 / SAMPLING_RATE)
    if len(t_target) < (WINDOW_SIZE_SEC * SAMPLING_RATE):
        return None
        
    resampled = {'timestamp_sec': t_target}
    for col in ['x-acceleration', 'y-acceleration', 'z-acceleration']:
        f = interp1d(t_raw, segment_df[col].values, kind='linear')
        resampled[col] = f(t_target)
    
    return pd.DataFrame(resampled)

processed_segments = []

print("Preprocessing: Resampling and gap handling...")
for (user, activity), group in df.groupby(['user', 'activity']):
    # Identify gaps > MAX_GAP_SEC
    # timestamp is in nanoseconds
    diffs = group['timestamp'].diff() / 1e9
    gap_indices = np.where(diffs > MAX_GAP_SEC)[0]
    
    # Split into segments
    start_idx = 0
    segments = []
    for gap_idx in gap_indices:
        segments.append(group.iloc[start_idx:gap_idx])
        start_idx = gap_idx
    segments.append(group.iloc[start_idx:])
    
    # Resample each segment
    for seg in segments:
        rs_seg = resample_segment(seg)
        if rs_seg is not None:
            rs_seg['user'] = user
            rs_seg['activity'] = activity
            processed_segments.append(rs_seg)

cleaned_df = pd.concat(processed_segments, ignore_index=True)

print("Normalizing axes and computing magnitude...")
# Normalize x, y, z (z-score normalization)
for col in ['x-acceleration', 'y-acceleration', 'z-acceleration']:
    cleaned_df[col] = (cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std()

cleaned_df['magnitude'] = np.sqrt(
    cleaned_df['x-acceleration']**2 + 
    cleaned_df['y-acceleration']**2 + 
    cleaned_df['z-acceleration']**2
)

print("Windowing and labeling...")
window_samples = int(WINDOW_SIZE_SEC * SAMPLING_RATE)
step_samples = int(window_samples * (1 - OVERLAP_PERCENT))

X = []
y = []
users = []

# Windowing within each resampled segment to avoid mixing users/activities
for _, segment in cleaned_df.groupby(['user', 'activity', (cleaned_df['timestamp_sec'].diff() > 1.0/SAMPLING_RATE + 0.001).cumsum()]):
    if len(segment) < window_samples:
        continue
        
    for i in range(0, len(segment) - window_samples + 1, step_samples):
        window = segment.iloc[i : i + window_samples]
        X.append(window[['x-acceleration', 'y-acceleration', 'z-acceleration', 'magnitude']].values)
        # Use majority activity (though it's constant per segment here)
        y.append(window['activity'].iloc[0])
        users.append(window['user'].iloc[0])

X = np.array(X)
y = np.array(y)
users = np.array(users)

print(f"Total windows: {len(X)}")

print("Creating subject-wise splits...")
# Train: 1-25, Val: 26-30, Test: 31-36
train_mask = np.isin(users, range(1, 26))
val_mask = np.isin(users, range(26, 31))
test_mask = np.isin(users, range(31, 37))

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

print(f"Saving processed dataset to {PROCESSED_DATA_PATH}...")
np.savez_compressed(
    PROCESSED_DATA_PATH, 
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test
)

# --- Cleaned-data visuals (mirror of one.py) ---
print("Generating cleaned-data visuals for User 33, Jogging...")
USER_ID = 33
ACTIVITY = 'Jogging'
TIME_SLICE_SECONDS = 10

# Filter cleaned_df for visual comparison
# Note: cleaned_df is already normalized and resampled
visual_df = cleaned_df[(cleaned_df['user'] == USER_ID) & (cleaned_df['activity'] == ACTIVITY)].copy()

if not visual_df.empty:
    # Pick the first continuous segment for plotting to mirror the "first N seconds" logic
    # Find gaps in timestamp_sec to identify segments
    visual_df['segment_id'] = (visual_df['timestamp_sec'].diff() > 1.0/SAMPLING_RATE + 0.001).cumsum()
    first_seg = visual_df[visual_df['segment_id'] == visual_df['segment_id'].iloc[0]].copy()
    
    # Re-zero time for the plot
    first_seg['time_rel'] = first_seg['timestamp_sec'] - first_seg['timestamp_sec'].min()
    plot_df = first_seg[first_seg['time_rel'] <= TIME_SLICE_SECONDS]

    # Plot 1: x-axis accel vs time
    plt.figure(figsize=(10, 6))
    plt.plot(plot_df['time_rel'], plot_df['x-acceleration'])
    plt.title(f'Cleaned WISDM – after preprocessing: X-Acceleration for User {USER_ID} ({ACTIVITY}) (First {TIME_SLICE_SECONDS}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('X-Acceleration (Normalized)')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Cleaned WISDM – after preprocessing_x_accel_vs_time.png'))
    plt.close()

    # Plot 2: magnitude vs time
    plt.figure(figsize=(10, 6))
    plt.plot(plot_df['time_rel'], plot_df['magnitude'])
    plt.title(f'Cleaned WISDM – after preprocessing: Acceleration Magnitude for User {USER_ID} ({ACTIVITY}) (First {TIME_SLICE_SECONDS}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration Magnitude (Normalized)')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Cleaned WISDM – after preprocessing_magnitude_vs_time.png'))
    plt.close()

    # Plot 3: sampling-interval histogram
    # Since it's resampled to 50Hz, this should be a sharp peak at 20ms
    plt.figure(figsize=(10, 6))
    sampling_intervals = visual_df['timestamp_sec'].diff().dropna() * 1000 # in ms
    # Filter out segment gaps for the histogram to see the resampled rate
    sampling_intervals = sampling_intervals[sampling_intervals < 500] 
    plt.hist(sampling_intervals, bins=50)
    plt.title(f'Cleaned WISDM – after preprocessing: Sampling Interval Histogram for User {USER_ID} ({ACTIVITY})')
    plt.xlabel('Sampling Interval (ms)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'Cleaned WISDM – after preprocessing_sampling_interval_histogram.png'))
    plt.close()

# Plot 4: activity count bar chart (cleaned dataset)
# Count by windows to be more representative of the final dataset
activity_counts = pd.Series(y).value_counts()

plt.figure(figsize=(10, 6))
activity_counts.plot(kind='bar')
plt.title('Cleaned WISDM – after preprocessing: Activity Count Bar Chart (Windows)')
plt.xlabel('Activity')
plt.ylabel('Number of Windows')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Cleaned WISDM – after preprocessing_activity_count_bar_chart.png'))
plt.close()

print(f"Generated cleaned visualization in '{OUTPUT_DIR}' directory.")
