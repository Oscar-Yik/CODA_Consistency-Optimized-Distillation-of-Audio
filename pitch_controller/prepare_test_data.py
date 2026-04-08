"""
One-time preprocessing script for the two test WAV files.
Run from inside pitch_controller/:  python prepare_test_data.py
Produces mel/, f0/, world/ .npy files alongside the vocal/ folder.
"""
import os
import sys
import numpy as np

# Add parent directory to path and import from root utils (not pitch_controller/utils.py)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import utils as root_utils
get_mel = root_utils.get_mel
get_f0 = root_utils.get_f0
get_world_mel = root_utils.get_world_mel

DATA_DIR = '../data/test_singer/'
VOCAL_DIR = os.path.join(DATA_DIR, 'vocal')


def save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


if __name__ == '__main__':
    files = [f for f in os.listdir(VOCAL_DIR) if f.endswith('.wav')]
    print(f'Found {len(files)} files: {files}')

    for fname in files:
        wav_path = os.path.join(VOCAL_DIR, fname)
        print(f'Processing {fname}...')

        # Use existing functions from utils.py - they expect file paths, not arrays
        save(os.path.join(DATA_DIR, 'mel',   fname + '.npy'), get_mel(wav_path))
        save(os.path.join(DATA_DIR, 'f0',    fname + '.npy'), get_f0(wav_path, method='pyin', padding=False))
        save(os.path.join(DATA_DIR, 'world', fname + '.npy'), get_world_mel(wav_path=wav_path))
        print(f'  saved mel / f0 / world')

    print('Done.')
