"""
Preprocessing script for OpenSinger dataset.
Run from inside pitch_controller/:  uv run prepare_test_data.py
Processes ManRaw and WomanRaw wav files and produces mel/, f0/, world/ .npy files in /data/training.
Also updates meta_fix.csv with processed files.
"""
import os
import sys
import numpy as np
import csv

# Add parent directory to path and import from root utils (not pitch_controller/utils.py)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import utils as root_utils
get_mel = root_utils.get_mel
get_f0 = root_utils.get_f0
get_world_mel = root_utils.get_world_mel

DATA_DIR = '../data/OpenSinger'
MAN_DIR = os.path.join(DATA_DIR, 'ManRaw')
WOMAN_DIR = os.path.join(DATA_DIR, 'WomanRaw')
TRAINING_DATA_DIR = '../data/training'
META_CSV = '../data/meta_fix.csv'

def save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def process_directory(source_dir, category, csv_writer, csvfile):
    """Process all wav files in a directory recursively."""
    wav_files = []
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if fname.endswith('.wav'):
                wav_files.append(os.path.join(root, fname))

    print(f'Found {len(wav_files)} wav files in {category}')

    for wav_path in wav_files:
        # Create a unique identifier from the relative path
        rel_path = os.path.relpath(wav_path, source_dir)
        # Remove .wav and replace path separators with underscores
        file_id = rel_path[:-4].replace(os.sep, '_')

        print(f'Processing {category}/{rel_path}...')

        try:
            # Save to training directory with category prefix
            base_name = f"{category}_{file_id}"
            save(os.path.join(TRAINING_DATA_DIR, 'mel',   base_name + '.npy'), get_mel(wav_path))
            save(os.path.join(TRAINING_DATA_DIR, 'f0',    base_name + '.npy'), get_f0(wav_path, method='pyin', padding=True))
            save(os.path.join(TRAINING_DATA_DIR, 'world', base_name + '.npy'), get_world_mel(wav_path=wav_path))
            print(f'  saved mel / f0 / world')

            # Write to CSV: subset, file_name, folder, subfolder
            # Note: 'vocal' in subfolder is a placeholder that gets replaced with 'mel'/'f0'/'world'
            csv_writer.writerow(['train', base_name, 'training/', 'vocal/'])
            csvfile.flush()  # Force write to disk immediately

        except Exception as e:
            print(f'  ERROR processing {wav_path}: {e}')
            continue


if __name__ == '__main__':
    print('Processing OpenSinger dataset...')

    # Initialize CSV file with header
    with open(META_CSV, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['subset', 'file_name', 'folder', 'subfolder'])
        csvfile.flush()  # Write header immediately

        if os.path.exists(MAN_DIR):
            process_directory(MAN_DIR, 'man', csv_writer, csvfile)
        else:
            print(f'Warning: {MAN_DIR} not found')

        if os.path.exists(WOMAN_DIR):
            process_directory(WOMAN_DIR, 'woman', csv_writer, csvfile)
        else:
            print(f'Warning: {WOMAN_DIR} not found')

    print('Done.')
