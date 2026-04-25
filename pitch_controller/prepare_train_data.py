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
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from tqdm import tqdm

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
TEST_DIR = '../data/test_samples'
TRAINING_DATA_DIR = '../data/training'
META_CSV = '../data/meta_fix.csv'

def save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def process_audio(wav_path, base_name, subset):
    # 1. Load the full audio
    wav, sr = librosa.load(wav_path, sr=24000)

    # 2. Split the audio on silence
    # top_db=30 cuts anything that is 30 decibels quieter than the loudest part of the track
    intervals = librosa.effects.split(wav, top_db=30)
    
    csv_rows = []
    
    # 3. Process each non-silent chunk separately
    for idx, (start, end) in enumerate(intervals):
        sub_wav = wav[start:end]
        
        # Skip chunks that are too short for your train_frames (128 frames)
        # 128 frames * ~256 hop_length = ~32,768 samples. We use 35,000 for safety.
        if len(sub_wav) < 35000:
            continue
            
        chunk_name = f"{base_name}_part{idx}"
        
        # Processing
        mel_data = root_utils.quick_get_mel(sub_wav)
        f0_data = root_utils.quick_get_f0(sub_wav, sr, method='world', padding=True)
        world_data = root_utils.quick_get_world_mel(sub_wav, sr)

        # Saving
        save(os.path.join(TRAINING_DATA_DIR, 'mel',   chunk_name + '.npy'), mel_data)
        save(os.path.join(TRAINING_DATA_DIR, 'f0',    chunk_name + '.npy'), f0_data)
        save(os.path.join(TRAINING_DATA_DIR, 'world', chunk_name + '.npy'), world_data)

        # Add this specific chunk to the CSV list
        csv_rows.append([subset, chunk_name, 'training/', 'vocal/'])
        
    return csv_rows

def process_test_file(wav_path, source_dir):
    """Worker function to split audio on silences, process parts, and return CSV data."""
    try:
        rel_path = os.path.relpath(wav_path, source_dir)
        file_id = rel_path[:-4].replace(os.sep, '_')
        base_name = f"test_{file_id}"

        csv_rows = process_audio(wav_path, base_name, "test")
        return csv_rows
        
    except Exception as e:
        return f"ERROR {wav_path}: {e}"

def process_single_file(wav_path, source_dir, category):
    """Worker function to split audio on silences, process parts, and return CSV data."""
    try:
        rel_path = os.path.relpath(wav_path, source_dir)
        file_id = rel_path[:-4].replace(os.sep, '_')
        base_name = f"{category}_{file_id}"

        csv_rows = process_audio(wav_path, base_name, "train")
        return csv_rows
        
    except Exception as e:
        return f"ERROR {wav_path}: {e}"
    
def multiprocessing(max_workers=4, subset="train"):
    # 1. Collect Tasks
    tasks = []

    if subset == "train":
        for source, cat in [(MAN_DIR, 'man'), (WOMAN_DIR, 'woman')]:
            if os.path.exists(source):
                for root, _, files in os.walk(source):
                    for fname in [f for f in files if f.endswith('.wav')]:
                        tasks.append((os.path.join(root, fname), source, cat))
    elif subset == "test": 
        for root, _, files in os.walk(TEST_DIR):
            for fname in [f for f in files if f.endswith('.wav')]:
                tasks.append((os.path.join(root, fname), TEST_DIR))

    if not tasks:
        print("No files found. Check your DATA_DIR paths.")
        sys.exit()

    # 2. Parallel Execution
    results_to_write = []
    print(f"Starting parallel processing of {len(tasks)} files...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Wrap the executor in tqdm for a progress bar
        if subset == "train":
            futures = [executor.submit(process_single_file, *t) for t in tasks]
        else:
            futures = [executor.submit(process_test_file, *t) for t in tasks]
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Audio", unit="file"):
            res = future.result()
            if isinstance(res, list):
                results_to_write.extend(res)
            else:
                tqdm.write(f"\n{res}")

    # 3. Write CSV at the end (safest for multi-process results)
    mode = "w" if subset == "train" else "a"
    with open(META_CSV, mode, newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if subset == "train":
            csv_writer.writerow(['subset', 'file_name', 'folder', 'subfolder'])
        csv_writer.writerows(results_to_write)

    print(f"Done. Processed {len(results_to_write)} files.")

if __name__ == '__main__':
    multiprocessing(max_workers=6, subset="test")
