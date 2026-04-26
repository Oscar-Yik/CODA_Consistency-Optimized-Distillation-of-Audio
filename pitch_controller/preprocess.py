import os
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def extract_and_rename_wavs(source_root, destination_folder):
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    source_path = Path(source_root)
    count = 0

    all_wavs = list(source_path.rglob('*.wav'))

    # rglob searches for .wav files recursively through all subdirectories
    for wav_file in tqdm(all_wavs, desc="Extracting Audio", unit="file"):
        # Get the name of the folder containing the wav file
        folder_name = wav_file.parent.name
        
        # Define the new filename: <folder_name>_<original_filename>
        new_filename = f"{folder_name}_{wav_file.name}"
        dest_path = Path(destination_folder) / new_filename
        
        # Copy the file to the new location
        shutil.copy2(wav_file, dest_path)
        count += 1

    print(f"\nDone! Successfully extracted {count} files to '{destination_folder}'.")

# --- Configuration ---
# Update these paths for your computer
DATASET_ROOT = '../data/raw/m4singer' 
OUTPUT_FOLDER = '../data/m4singer'

if __name__ == "__main__":
    extract_and_rename_wavs(DATASET_ROOT, OUTPUT_FOLDER)
