import os
import csv
import numpy as np
from tqdm import tqdm

DATA_DIR = '../data/'
META_CSV = os.path.join(DATA_DIR, 'meta_fix.csv')
CLEAN_CSV = os.path.join(DATA_DIR, 'meta_clean.csv')

# Thresholds (You can tune these)
# If the maximum volume in the entire clip never exceeds this, it's considered dead air.
SILENCE_THRESHOLD = -4.0 

def clean_dataset():
    print(f"Scanning {META_CSV} for silent audio clips...")
    
    valid_rows = []
    silent_count = 0
    missing_count = 0
    
    with open(META_CSV, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        valid_rows.append(header)
        
        # Read all rows into a list so we can use tqdm
        rows = list(reader)
        
    for row in tqdm(rows):
        subset, file_name, folder, subfolder = row
        
        # Construct path to the mel-spectrogram .npy file
        # row['subfolder'] is 'vocal/', but the arrays are saved in 'mel/'
        mel_path = os.path.join(DATA_DIR, folder, 'mel', file_name + '.npy')
        
        if not os.path.exists(mel_path):
            missing_count += 1
            continue
            
        try:
            # Load the mel-spectrogram
            mel = np.load(mel_path)
            
            # Check the maximum value. If the max value is very low, the whole clip is silent.
            max_val = np.max(mel)
            
            if max_val < SILENCE_THRESHOLD:
                silent_count += 1
                # We skip adding this to valid_rows
            else:
                valid_rows.append(row)
                
        except Exception as e:
            print(f"Error loading {mel_path}: {e}")
            missing_count += 1

    # Write the clean data to a new CSV
    with open(CLEAN_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(valid_rows)
        
    print("\n--- Cleanup Complete ---")
    print(f"Total Original Files: {len(rows)}")
    print(f"Silent Files Removed: {silent_count}")
    print(f"Missing Files Skipped: {missing_count}")
    print(f"Total Valid Files Remaining: {len(valid_rows) - 1}") # -1 for header
    print(f"Saved clean list to: {CLEAN_CSV}")

if __name__ == "__main__":
    clean_dataset()