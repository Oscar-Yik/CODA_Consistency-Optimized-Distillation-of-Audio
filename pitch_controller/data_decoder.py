import os
import json
import yaml
import random
import numpy as np
import torch
from types import SimpleNamespace

from modules.BigVGAN.inference import load_model
from utils import save_audio

def main():
    # 1. Load Configurations
    with open("config.json", "r") as f:
        args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    mel_cfg = config['logmel']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. Initialize Vocoder
    print('Loading Vocoder...')
    hifigan, _ = load_model(args.vocoder_dir, device=device)

    # Output directory
    out_dir = "verify_chunks_audio"
    os.makedirs(out_dir, exist_ok=True)

    # 3. Locate the Mel-Spectrogram files
    mel_dir = os.path.join(args.data_dir, 'training', 'mel')
    if not os.path.exists(mel_dir):
        print(f"Error: Could not find {mel_dir}")
        return
        
    all_mel_files = [f for f in os.listdir(mel_dir) if f.endswith('.npy')]
    
    if not all_mel_files:
        print(f"No .npy files found in {mel_dir}. Run your preprocessing script first!")
        return

    # Pick 10 random files to test
    num_tests = 10
    # test_files = random.sample(all_mel_files, min(num_tests, len(all_mel_files)))
    test_files = all_mel_files[:num_tests]

    print(f"Testing {len(test_files)} random chunked files...\n")

    with torch.no_grad():
        for i, file_name in enumerate(test_files):
            mel_path = os.path.join(mel_dir, file_name)
            
            # Load the raw numpy array
            mel_np = np.load(mel_path)
            
            # BigVGAN expects shape [Batch, Channels, Time]. 
            # Our numpy arrays are [Channels, Time], so we add the Batch dimension.
            mel_tensor = torch.from_numpy(mel_np).float().unsqueeze(0).to(device)
            
            # Pass directly through the vocoder (no normalization needed for the raw .npy!)
            audio = hifigan(mel_tensor)
            
            # Save to wav
            out_path = os.path.join(out_dir, f"chunk_{i}_{file_name.replace('.npy', '.wav')}")
            save_audio(out_path, mel_cfg['sampling_rate'], audio)
            print(f"Saved: {out_path}")

    print(f"\nDone! Listen to the files in the '{out_dir}' folder.")

if __name__ == "__main__":
    main()