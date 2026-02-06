
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the path to the dataset
DATA_PATH = 'wisdm_dataset/WISDM_ar_v1.1_raw.txt'
OUTPUT_DIR = 'raw_visuals'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column names based on the raw_about.txt
COLUMN_NAMES = [
    'user', 'activity', 'timestamp', 
    'x-acceleration', 'y-acceleration', 'z-acceleration'
]

# Load the dataset
# The dataset has a semicolon at the end of each line, so we need to handle that.
try:
    df = pd.read_csv(
        DATA_PATH, 
        header=None, 
        names=COLUMN_NAMES, 
        sep=',', 
        decimal='.', 
        on_bad_lines='skip'
    )
    
    # Remove the trailing semicolon from the last column if present
    df['z-acceleration'] = df['z-acceleration'].astype(str).str.replace(';', '')
    df['z-acceleration'] = pd.to_numeric(df['z-acceleration'])

except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure the data file is correctly formatted and accessible.")
    exit()

# --- 1. Load only a small slice + Pick 1 user + 1 activity (e.g., user 33, Jogging). ---
USER_ID = 33
ACTIVITY = 'Jogging'
TIME_SLICE_SECONDS = 10 # 10 seconds

user_activity_df = df[(df['user'] == USER_ID) & (df['activity'] == ACTIVITY)].copy()

if user_activity_df.empty:
    print(f"No data found for User {USER_ID} performing {ACTIVITY}.")
    exit()

# Convert timestamp to seconds for easier slicing (original is nanoseconds)
user_activity_df['timestamp_sec'] = (user_activity_df['timestamp'] - user_activity_df['timestamp'].min()) / 1_000_000_000

# Select the first N seconds
sliced_df = user_activity_df[user_activity_df['timestamp_sec'] <= TIME_SLICE_SECONDS]

if sliced_df.empty:
    print(f"No data found for the first {TIME_SLICE_SECONDS} seconds for User {USER_ID} performing {ACTIVITY}.")
    exit()

# --- 2. Make only 4 plots (from that slice): ---

# Calculate magnitude of acceleration for the sliced data
sliced_df['magnitude'] = np.sqrt(
    sliced_df['x-acceleration']**2 + 
    sliced_df['y-acceleration']**2 + 
    sliced_df['z-acceleration']**2
)

# Plot 1: x-axis accel vs time
plt.figure(figsize=(10, 6))
plt.plot(sliced_df['timestamp_sec'], sliced_df['x-acceleration'])
plt.title(f'Raw WISDM – before preprocessing: X-Acceleration for User {USER_ID} ({ACTIVITY}) (First {TIME_SLICE_SECONDS}s)')
plt.xlabel('Time (s)')
plt.ylabel('X-Acceleration')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'Raw WISDM – before preprocessing_x_accel_vs_time.png'))
plt.close()

# Plot 2: magnitude vs time
plt.figure(figsize=(10, 6))
plt.plot(sliced_df['timestamp_sec'], sliced_df['magnitude'])
plt.title(f'Raw WISDM – before preprocessing: Acceleration Magnitude for User {USER_ID} ({ACTIVITY}) (First {TIME_SLICE_SECONDS}s)')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration Magnitude')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'Raw WISDM – before preprocessing_magnitude_vs_time.png'))
plt.close()

# Plot 3: sampling-interval histogram (∆timestamp)
# Calculate sampling intervals for the sliced data
sliced_df['delta_timestamp'] = sliced_df['timestamp'].diff() / 1_000_000 # convert nanoseconds to milliseconds

# Drop the first NaN value from diff()
sampling_intervals = sliced_df['delta_timestamp'].dropna()

plt.figure(figsize=(10, 6))
plt.hist(sampling_intervals, bins=50)
plt.title(f'Raw WISDM – before preprocessing: Sampling Interval Histogram for User {USER_ID} ({ACTIVITY}) (First {TIME_SLICE_SECONDS}s)')
plt.xlabel('Sampling Interval (ms)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'Raw WISDM – before preprocessing_sampling_interval_histogram.png'))
plt.close()

# Plot 4: activity count bar chart (whole dataset is OK for this one)
activity_counts = df['activity'].value_counts()

plt.figure(figsize=(10, 6))
activity_counts.plot(kind='bar')
plt.title('Raw WISDM – before preprocessing: Activity Count Bar Chart (Whole Dataset)')
plt.xlabel('Activity')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Raw WISDM – before preprocessing_activity_count_bar_chart.png'))
plt.close()

print(f"Generated raw data visualizations in the '{OUTPUT_DIR}' directory.")
print("You can find the plots:")
print(f"- {OUTPUT_DIR}/Raw WISDM – before preprocessing_x_accel_vs_time.png")
print(f"- {OUTPUT_DIR}/Raw WISDM – before preprocessing_magnitude_vs_time.png")
print(f"- {OUTPUT_DIR}/Raw WISDM – before preprocessing_sampling_interval_histogram.png")
print(f"- {OUTPUT_DIR}/Raw WISDM – before preprocessing_activity_count_bar_chart.png")

