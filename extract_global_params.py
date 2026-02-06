import pandas as pd
import numpy as np
import json
import os

DATA_PATH = 'wisdm_dataset/WISDM_ar_v1.1_raw.txt'
COLUMN_NAMES = ['user', 'activity', 'timestamp', 'x-accel', 'y-accel', 'z-accel']

def main():
    print("Loading raw data for parameter extraction...")
    try:
        df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES, on_bad_lines='skip')
        df['z-accel'] = pd.to_numeric(df['z-accel'].astype(str).str.replace(';', ''), errors='coerce')
        df.dropna(inplace=True)
        
        # Calculate standard deviation for x, y, z axes
        global_std = {
            'x': float(df['x-accel'].std()),
            'y': float(df['y-accel'].std()),
            'z': float(df['z-accel'].std())
        }
        
        with open('backend/global_std.json', 'w') as f:
            json.dump(global_std, f, indent=2)
        
        # Save labels for metadata
        labels = sorted(df['activity'].unique().tolist())
        with open('backend/model_metadata.json', 'w') as f:
            json.dump({'labels': labels, 'model_type': 'rf'}, f, indent=2)
            
        print(f"Extraction complete. Parameters saved to backend/global_std.json and backend/model_metadata.json")
        print(f"Global STD: {global_std}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
