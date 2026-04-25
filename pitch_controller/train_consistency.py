import os, yaml, csv, json, shutil
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from diffusers import DDIMScheduler

from dataset import VCDecLPCDataset, VCDecLPCBatchCollate
from models.unet import UNetPitcher
from modules.BigVGAN.inference import load_model
from cd_trainer import ConsistencyTrainer
from utils import save_plot, save_audio, minmax_norm_diff, reverse_minmax_norm_diff, calculate_f0_error

import matplotlib.pyplot as plt


with open("config.json", "r") as f:
    args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

args.save_ori = True
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
mel_cfg = config['logmel']
ddpm_cfg = config['ddpm']
unet_cfg = config['unet']
f0_type = "log"

def graph_val_mel_spectrograms(gt_mel_norm, pred, epoch, fig_path):
    _, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Ground truth
    axes[0].imshow(gt_mel_norm[0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Mel Frequency')

    # Prediction
    axes[1].imshow(pred[0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title(f'Prediction (Epoch {epoch})')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Mel Frequency')

    # Difference
    diff = (pred[0] - gt_mel_norm[0]).abs().cpu().numpy()
    im = axes[2].imshow(diff, aspect='auto', origin='lower', cmap='hot')
    axes[2].set_title(f'Absolute Difference (Mean: {diff.mean():.4f})')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Mel Frequency')
    plt.colorbar(im, ax=axes[2], label='|Difference|')

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

def persist_config(log_dir):
    files_to_persist = [
        "config.json",
        "train_consistency.py",
        "cd_trainer.py",
        "models/consistency.py"
    ]

    paths_to_persist = [Path(p) for p in files_to_persist]

    for path_to_persist in paths_to_persist:
        shutil.copy(path_to_persist, f"{log_dir}/{path_to_persist.name}")

if __name__ == "__main__":
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        args.device = 'cuda'
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
    else:
       args.device = 'cpu'
    print(f"Using device: {args.device}")

    if os.path.exists(args.log_dir) is False:
        os.makedirs(args.log_dir)

    if os.path.exists(args.ckpt_dir) is False:
        os.makedirs(args.ckpt_dir)

    print('Initializing vocoder...')
    hifigan, cfg = load_model(args.vocoder_dir, device='cpu')

    print('Initializing data loaders...')
    train_set = VCDecLPCDataset(args.data_dir, subset='train', content_dir=args.lpc_dir, f0_type=f0_type)

    # Overfit small sample stuff
    if args.overfit_one_sample:
        print("[Override] Overfit on small sample")
        overfit_indices = list(range(args.num_overfit_samples)) 
        train_set = Subset(train_set, overfit_indices)

    collate_fn = VCDecLPCBatchCollate(args.train_frames)

    if args.num_workers == 0:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                collate_fn=collate_fn, num_workers=0, drop_last=False)
    else: 
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True,
                              pin_memory=True, prefetch_factor=2)

    print('Loading Teacher and Initializing Consistency Trainer...')
    teacher_model = UNetPitcher(**unet_cfg)
    unet_path = '../ckpts/world_fixed_40.pt'
    state_dict = torch.load(unet_path, weights_only=True) # suppress security warning
    for key in list(state_dict.keys()):
        state_dict[key.replace('_orig_mod.', '')] = state_dict.pop(key)
    teacher_model.load_state_dict(state_dict)
    
    trainer = ConsistencyTrainer(
        teacher_model=teacher_model,
        unet_cfg=unet_cfg,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ema_mu=args.ema_mu,
        mixed_precision=args.mixed_precision,
    )

    # --- NEW: RESUME TRAINING LOGIC ---
    start_epoch = 1
    if hasattr(args, 'resume_ckpt') and args.resume_ckpt:
        if os.path.exists(args.ckpt_file):
            print(f"Resuming training from checkpoint: {args.ckpt_file}")
            state_dict = torch.load(args.ckpt_file, map_location=args.device, weights_only=True)
            
            # Load into BOTH models to reset the EMA baseline
            trainer.student.load_state_dict(state_dict)
            trainer.target_student.load_state_dict(state_dict)
            
            start_epoch = getattr(args, 'resume_epoch', 0) + 1
            print(f"Resuming at Epoch {start_epoch}...")
        else:
            print(f"WARNING: Checkpoint {args.ckpt_file} not found. Starting from scratch.")
    # ----------------------------------

    # prepare DPM scheduler
    noise_scheduler = DDIMScheduler(num_train_timesteps=ddpm_cfg['num_train_steps'])
    noise_scheduler.set_timesteps(ddpm_cfg['inference_steps']) 

    print('Start Consistency Distillation.')
    global_step = 0

    val_batch = next(iter(train_loader))
    gt_mel = val_batch['mel1'][0:1].to(args.device)
    gt_mel_norm = minmax_norm_diff(gt_mel, vmax=mel_cfg['max'], vmin=mel_cfg['min'])

    val_mean = val_batch['content1'][0:1].to(args.device)
    val_mean_norm = minmax_norm_diff(val_mean, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
    val_f0 = val_batch['f0_1'][0:1].to(args.device)

    # --- NEW: Save the Validation Target Audio ONCE before training starts ---
    val_target_path = f'{args.log_dir}/val_target_audio.wav'
    with torch.no_grad():
        audio_target_val = hifigan(gt_mel.to('cpu'))
    save_audio(val_target_path, mel_cfg['sampling_rate'], audio_target_val)
    # -------------------------------------------------------------------------

    # -- 2. NEW Evaluation Metrics Log --
    eval_log_path = f'{args.log_dir}/eval_metrics.csv'
    with open(eval_log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'mel_mse', 'f0_mae', 'blur_ratio', 'mel_mse_full'])

    # -- 3. Save all config to log_dir --
    persist_config(args.log_dir)

    for epoch in range(start_epoch, args.epochs + 1):

        trainer.student.train()
        losses = []

        if epoch % args.save_every == 0:
            print(f'Epoch: {epoch} [iteration: {global_step}]')

        loader = train_loader if args.overfit_one_sample else tqdm(train_loader)
        for step, batch in enumerate(loader):
            # make spectrogram range from -1 to 1
            mel = batch['mel1'].to(args.device)
            mel = minmax_norm_diff(mel, vmax=mel_cfg['max'], vmin=mel_cfg['min'])

            if unet_cfg["use_ref_t"]:
                mel_ref = batch['mel2'].to(args.device)
                mel_ref = minmax_norm_diff(mel_ref, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
            else:
                mel_ref = None

            f0 = batch['f0_1'].to(args.device)

            mean = batch['content1'].to(args.device)
            mean = minmax_norm_diff(mean, vmax=mel_cfg['max'], vmin=mel_cfg['min'])

            # Randomly sample a timestep index from our set of 50
            max_t_idx = len(noise_scheduler.timesteps) - 1
            # current_min_t = min(max_t_idx_curriculum, max_t_idx)

            t_idx = torch.randint(0, max_t_idx + 1, (1,)).item()
            loss = trainer.train_step(mel, mean, f0, noise_scheduler, t_idx)

            global_step += 1

        if epoch % args.save_every > 0:
            continue
        
        if not args.overfit_one_sample:
            print('Saving model...\n')
            torch.save(trainer.target_student.state_dict(), f"{args.ckpt_dir}/consistency_model_{epoch}.pt")

        print('Inference and Evaluation...\n')
        trainer.target_student.eval()
        with torch.no_grad():
            # 2. Run Student (1 Step from pure noise)
            noise = torch.randn_like(val_mean_norm)
            t_max = torch.tensor([noise_scheduler.timesteps[50]], device=args.device)
            noise_input = noise_scheduler.add_noise(gt_mel_norm[0:1], noise, t_max)
            pred = trainer.target_student(noise_input, t_max, val_mean_norm, val_f0, noise_scheduler)
            
            # 3. Calculate Mel-Spectrogram MSE
            mel_mse = F.mse_loss(pred, gt_mel_norm)

            # 4. Rescale back to normal audio ranges and save
            pred_rescaled = reverse_minmax_norm_diff(pred, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
            audio_student = hifigan(pred_rescaled.to('cpu'))
            student_wav_path = f'{args.log_dir}/ep{epoch}_student_audio.wav'
            save_audio(student_wav_path, mel_cfg['sampling_rate'], audio_student)

            # 5. Save Spectrogram Plots (Visual Check)
            fig_path = f'{args.log_dir}/mel_comparison_ep{epoch}.png'
            graph_val_mel_spectrograms(gt_mel_norm, pred, epoch, fig_path)




            # -- REPEAT THE SAME STEPS BUT FOR FULL MODEL
            t_max_full = torch.tensor([noise_scheduler.timesteps[0]], device=args.device)
            noise_input_full = noise_scheduler.add_noise(gt_mel_norm[0:1], noise, t_max_full)
            pred_full = trainer.target_student(noise_input_full, t_max_full, val_mean_norm, val_f0, noise_scheduler)

            mel_mse_full = F.mse_loss(pred_full, gt_mel_norm)
            print(f"--> Epoch {epoch} | Validation Mel MSE: {mel_mse.item():.6f} | Validation Mel MSE Full: {mel_mse_full.item():.6f}")

            pred_rescaled_full = reverse_minmax_norm_diff(pred_full, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
            audio_student_full = hifigan(pred_rescaled_full.to('cpu'))
            student_wav_path_full = f'{args.log_dir}/ep{epoch}_student_audio_full.wav'
            save_audio(student_wav_path_full, mel_cfg['sampling_rate'], audio_student_full)

            fig_path = f'{args.log_dir}/mel_comparison_ep{epoch}_full.png'
            graph_val_mel_spectrograms(gt_mel_norm, pred_full, epoch, fig_path)



            
            

                
            # 7. Calculate Pitch Tracking Error
            # We use the actual saved wavs so we measure exactly what the vocoder produced
            # If it's not the first epoch, compare to the ep2 ground truth we saved earlier
            f0_mae_val = None

            if os.path.exists(val_target_path):
                f0_mae_val = calculate_f0_error(student_wav_path, val_target_path, sr=mel_cfg['sampling_rate'])
                print(f"--> Epoch {epoch} | F0 Tracking Error (MAE): {f0_mae_val:.2f} Hz\n")
            
                # The Variance Ratio (Sharpness Tracker)
                # gt_mel_norm.var() is the "true" sharpness. pred.var() is your model's sharpness.
                pred_var = pred.var().item()
                gt_var = gt_mel_norm.var().item()
                variance_ratio = pred_var / gt_var
                print(f"--> Epoch {epoch} | Blur Check (Variance Ratio): {variance_ratio:.4f} (Target: ~1.0)")

            # -- Append to Eval CSV --
            with open(eval_log_path, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, mel_mse.item(), f0_mae_val, variance_ratio, mel_mse_full])
