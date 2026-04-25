import os
import yaml
import json
import shutil
from tqdm import tqdm
import numpy as np
import librosa 
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler

from dataset import VCDecLPCDataset, VCDecLPCBatchCollate
from models.unet import UNetPitcher
from models.consistency import ConsistencyPitcher
from modules.BigVGAN.inference import load_model
from utils import save_audio, save_plot, minmax_norm_diff, reverse_minmax_norm_diff


"""
Usage:
Move checkpoint file from 04-21-2026_full-run/consistency_model_8.pt in google drive to pitch_controller/ckpt_consistency/
Run this command in terminal in pitch_controller/ `uv run test_consistency`
"""
def main():

    with open("test_config.json", "r") as f:
        cmd_args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    with open(cmd_args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    mel_cfg = config['logmel']
    ddpm_cfg = config['ddpm']
    unet_cfg = config['unet']
    f0_type = "log"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(cmd_args.out_dir, exist_ok=True)

    # 2. Initialize Vocoder
    print('Loading Vocoder...')
    hifigan, _ = load_model(cmd_args.vocoder_dir, device=device) 

    # 3. Load the Student Model
    print(f'Loading Student Model from {cmd_args.ckpt}...')
    unet = UNetPitcher(**unet_cfg).to(device)
    student = ConsistencyPitcher(unet).to(device)
    
    state_dict = torch.load(cmd_args.ckpt, map_location=device, weights_only=True)
    student.load_state_dict(state_dict)
    student.eval()

    # 4. Set up the Scheduler
    noise_scheduler = DDIMScheduler(num_train_timesteps=ddpm_cfg['num_train_steps'])
    noise_scheduler.set_timesteps(ddpm_cfg['inference_steps'])
    t_idx = cmd_args.t_idx
    t_max = torch.as_tensor(noise_scheduler.timesteps[t_idx]).to(device)
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    # 5. Load Training Dataset (Sanity Check)
    print('Loading Training Data for Sanity Check...')
    train_set = VCDecLPCDataset(cmd_args.data_dir, subset='test', content_dir=cmd_args.lpc_dir, f0_type=f0_type)
    collate_fn = VCDecLPCBatchCollate(cmd_args.train_frames)
    
    # shuffle=False ensures we grab the same files every time for consistent debugging
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    shutil.copy("test_config.json", f"{cmd_args.out_dir}/test_config.json")

    print(f'Starting generation for {cmd_args.num_samples} samples...')
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader, total=cmd_args.num_samples)):
            if i >= cmd_args.num_samples:
                break

            # Extract inputs
            gt_mel = batch['mel1'].to(device)
            content = batch['content1'].to(device)
            f0 = batch['f0_1'].to(device)

            # Normalize exactly like training
            content_norm = minmax_norm_diff(content, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
            
            # --- THE GROUND TRUTH VOCODER TEST ---
            # We reverse the normalization on the RAW dataloader mel.
            # If this produces silence, your dataset/vocoder is broken.
            audio_target = hifigan(gt_mel.to(device))
            save_audio(f'{cmd_args.out_dir}/sample_{i}_target.wav', mel_cfg['sampling_rate'], audio_target)
            save_plot(gt_mel.squeeze().cpu(), f'{cmd_args.out_dir}/sample_{i}_target_mel.png')


            # --- THE MODEL TEST ---
            # gt_mel_norm = minmax_norm_diff(gt_mel, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
            # noise = torch.randn_like(content_norm)
            # noisy_input = noise_scheduler.add_noise(gt_mel_norm, noise, t_max)
            # pred = student(noisy_input, t_max, content_norm, f0, noise_scheduler)



            chain_indices = list(range(0, 100, cmd_args.chain_steps))
            current_input = torch.randn_like(content_norm)
            
            for step_num, current_t_idx in enumerate(chain_indices):
                t_current = torch.as_tensor([noise_scheduler.timesteps[current_t_idx]], device=device)
                pred_x0 = student(current_input, t_current, content_norm, f0, noise_scheduler)
                
                if step_num < len(chain_indices) - 1:
                    next_t_idx = chain_indices[step_num + 1]
                    t_next = torch.as_tensor([noise_scheduler.timesteps[next_t_idx]], device=device)
                    fresh_noise = torch.randn_like(pred_x0)
                    current_input = noise_scheduler.add_noise(pred_x0, fresh_noise, t_next)
                else:
                    pred = pred_x0



            pred_rescaled = reverse_minmax_norm_diff(pred, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
            audio_student = hifigan(pred_rescaled)
            
            save_audio(f'{cmd_args.out_dir}/sample_{i}_student.wav', mel_cfg['sampling_rate'], audio_student)
            save_plot(pred_rescaled.squeeze().cpu(), f'{cmd_args.out_dir}/sample_{i}_student_mel.png')

    print(f"\nDone! Check the ./{cmd_args.out_dir}/ folder.")

# This is almost exactly the same as the above but I don't want to spend brain power to combine them
def test_teacher(): 
    # --- 1. CONFIG & SETUP ---
    with open("config.json", "r") as f:
        args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    mel_cfg = config['logmel']
    ddpm_cfg = config['ddpm']
    unet_cfg = config['unet']
    f0_type = "log"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    out_dir = "teacher_eval_output"
    os.makedirs(out_dir, exist_ok=True)

    # --- 2. LOAD VOCODER ---
    print('Loading HiFi-GAN Vocoder...')
    hifigan, _ = load_model(args.vocoder_dir, device='cpu')

    # --- 3. LOAD DATASET (Just 1 batch) ---
    print('Loading a test sample...')
    train_set = VCDecLPCDataset(args.data_dir, subset='train', content_dir=args.lpc_dir, f0_type=f0_type)
    collate_fn = VCDecLPCBatchCollate(args.train_frames)

    loader = DataLoader(train_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))

    # Prepare conditionings
    mean = batch['content1'].to(device)
    mean_norm = minmax_norm_diff(mean, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
    f0 = batch['f0_1'].to(device)
    gt_mel = batch['mel1'].to(device)

    # --- 4. LOAD TEACHER MODEL ---
    print('Loading Teacher UNet...')
    teacher = UNetPitcher(**unet_cfg).to(device)
    
    # Load weights, stripping DDP/compile prefixes if they exist
    state_dict = torch.load('../ckpts/world_fixed_40.pt', map_location=device, weights_only=True)
    for key in list(state_dict.keys()):
        state_dict[key.replace('_orig_mod.', '')] = state_dict.pop(key)
        
    teacher.load_state_dict(state_dict)
    teacher.eval()

    # --- 5. DIFFUSION SAMPLING LOOP ---
    print('Initializing DDIM Scheduler...')
    noise_scheduler = DDIMScheduler(num_train_timesteps=ddpm_cfg['num_train_steps'])
    noise_scheduler.set_timesteps(ddpm_cfg['inference_steps'])
    
    # Start with pure noise of the same shape as the expected spectrogram
    x_t = torch.randn_like(mean_norm).to(device)
    
    print(f'Starting Reverse Diffusion ({ddpm_cfg["inference_steps"]} steps)...')
    with torch.no_grad():
        for t in tqdm(noise_scheduler.timesteps):
            # Create a batched timestep tensor
            t_batch = torch.tensor([t.item()] * x_t.shape[0], device=device)
            
            # Predict noise/velocity
            model_output = teacher(x=x_t, mean=mean_norm, f0=f0, t=t_batch)
            
            # Take a step backward in time (denoise slightly)
            x_t = noise_scheduler.step(model_output, t.item(), x_t).prev_sample

    # --- 6. POST-PROCESSING & SAVING ---
    print('Generation complete. Processing audio and plots...')
    
    # The final x_0 is our predicted mel-spectrogram
    pred_mel_norm = x_t
    
    # Reverse normalization to get back to target scale (e.g., -10 to 0)
    pred_mel = reverse_minmax_norm_diff(pred_mel_norm, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
    
    # Save Heatmaps
    save_plot(pred_mel.squeeze().cpu(), f'{out_dir}/teacher_predicted_mel.png')
    save_plot(gt_mel.squeeze().cpu(), f'{out_dir}/ground_truth_mel.png')
    
    # Generate and Save Audio
    audio_teacher = hifigan(pred_mel.to('cpu'))
    save_audio(f'{out_dir}/teacher_audio.wav', mel_cfg['sampling_rate'], audio_teacher)
    
    audio_gt = hifigan(gt_mel.to('cpu'))
    save_audio(f'{out_dir}/ground_truth_audio.wav', mel_cfg['sampling_rate'], audio_gt)

    print(f"Done! Check the '{out_dir}' folder for the results.")

if __name__ == "__main__":
    main()
    # test_teacher()