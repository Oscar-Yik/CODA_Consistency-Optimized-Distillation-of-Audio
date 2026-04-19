import os, yaml, csv, json, time
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

from diffusers import DDIMScheduler

from dataset import VCDecLPCDataset, VCDecLPCBatchCollate
from models.unet import UNetPitcher
from modules.BigVGAN.inference import load_model
from cd_trainer import ConsistencyTrainer
from utils import save_plot, save_audio, minmax_norm_diff, reverse_minmax_norm_diff, calculate_f0_error


with open("config.json", "r") as f:
    args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

args.save_ori = True
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
mel_cfg = config['logmel']
ddpm_cfg = config['ddpm']
unet_cfg = config['unet']
f0_type = unet_cfg.get('pitch_type', 'bins')

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

    if os.path.exists(args.log_dir) is False:
        os.makedirs(args.log_dir)

    if os.path.exists(args.ckpt_dir) is False:
        os.makedirs(args.ckpt_dir)

    print('Initializing vocoder...')
    hifigan, cfg = load_model(args.vocoder_dir, device='cpu')

    print('Initializing data loaders...')
    train_set = VCDecLPCDataset(args.data_dir, subset='train', content_dir=args.lpc_dir, f0_type=f0_type)



    # Partition dataset
    # print("WARNING: Using Toy Dataset (17232 samples / ~2154 batches)")
    # # Grab the first 800 indices from the dataset
    # toy_indices = list(range(17232)) 
    # train_set = Subset(train_set, toy_indices)

    # Overfit 1 sample stuff
    if args.overfit_one_sample:
        print("[Override] Overfit on one sample")
        overfit_indices = list(range(args.batch_size)) 
        train_set = Subset(train_set, overfit_indices)




    collate_fn = VCDecLPCBatchCollate(args.train_frames)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True,
                              pin_memory=True, prefetch_factor=2)

    # val_set = VCDecLPCTest(args.data_dir, content_dir=args.lpc_dir, f0_type=f0_type)
    # val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

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
        weight_decay=args.weight_decay
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

    # --- NEW: Save the Validation Target Audio ONCE before training starts ---
    val_target_path = f'{args.log_dir}/val_target_audio.wav'
    gt_mel_val = val_batch['mel1'][0:1]
    audio_target_val = hifigan(gt_mel_val.to('cpu'))
    save_audio(val_target_path, mel_cfg['sampling_rate'], audio_target_val)
    # -------------------------------------------------------------------------

    # -- 1. Existing Trajectory Log --
    traj_log_path = f'{args.log_dir}/trajectory_loss_{time.time()}.csv'
    with open(traj_log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['global_step', 'epoch', 't_idx', 'loss'])

    # -- 2. NEW Evaluation Metrics Log --
    eval_log_path = f'{args.log_dir}/eval_metrics_{time.time()}.csv'
    with open(eval_log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'mel_mse', 'f0_mae'])

    if args.overfit_one_sample:
        overfit_batch = val_batch

        # Pre-normalize and move to GPU
        of_mel = overfit_batch['mel1'].to(args.device)
        of_mel = minmax_norm_diff(of_mel, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
        of_f0 = overfit_batch['f0_1'].to(args.device)
        of_mean = overfit_batch['content1'].to(args.device)
        of_mean = minmax_norm_diff(of_mean, vmax=mel_cfg['max'], vmin=mel_cfg['min'])

        mel = of_mel
        f0 = of_f0
        mean = of_mean 

    for epoch in range(start_epoch, args.epochs + 1):
        print(f'Epoch: {epoch} [iteration: {global_step}]')
        trainer.student.train()
        losses = []
        trajectory_data = []

        if args.overfit_one_sample:
            t_idx = torch.randint(0, len(noise_scheduler.timesteps) - 1, (1,)).item()
            loss = trainer.train_step(mel, mean, f0, noise_scheduler, t_idx)
            trajectory_data.append((global_step, epoch, t_idx, loss))
            global_step += 1
        else:
            for step, batch in enumerate(tqdm(train_loader)):
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
                t_idx = torch.randint(0, len(noise_scheduler.timesteps) - 1, (1,)).item()
                loss = trainer.train_step(mel, mean, f0, noise_scheduler, t_idx)

                trajectory_data.append((global_step, epoch, t_idx, loss))
                global_step += 1

                # logging
                if global_step % args.log_step == 0:
                    with open(traj_log_path, 'a', newline='') as f:
                        csv.writer(f).writerows(trajectory_data)
                    trajectory_data = []

        if epoch % args.save_every > 0:
            continue

        print('Saving model...\n')
        torch.save(trainer.target_student.state_dict(), f"{args.ckpt_dir}/consistency_model_{epoch}.pt")

        print('Save trajectory data')
        with open(traj_log_path, 'a', newline='') as f:
            csv.writer(f).writerows(trajectory_data)
        trajectory_data = []

        print('Inference and Evaluation...\n')
        trainer.target_student.eval()
        with torch.no_grad():
            # 1. Grab the Ground Truth from the very last batch of the epoch
            gt_mel = val_batch['mel1'][0:1].to(args.device)
            test_mean = val_batch['content1'][0:1].to(args.device)
            test_f0 = val_batch['f0_1'][0:1].to(args.device)

            test_mean_norm = minmax_norm_diff(test_mean, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
            gt_mel_norm = minmax_norm_diff(gt_mel, vmax=mel_cfg['max'], vmin=mel_cfg['min'])

            # 2. Run Student (1 Step from pure noise)
            noise = torch.randn_like(test_mean_norm)
            t_max = torch.tensor([noise_scheduler.timesteps[0]], device=args.device)
            pred = trainer.target_student(noise, t_max, test_mean_norm, test_f0, noise_scheduler)
            
            # 3. Calculate Mel-Spectrogram MSE
            mel_mse = F.mse_loss(pred, gt_mel_norm)
            print(f"--> Epoch {epoch} | Validation Mel MSE: {mel_mse.item():.6f}")

            # 4. Rescale back to normal audio ranges
            pred_rescaled = reverse_minmax_norm_diff(pred, vmax=mel_cfg['max'], vmin=mel_cfg['min'])

            # 5. Save Spectrogram Plots (Visual Check)
            save_plot(pred_rescaled.squeeze().cpu(), f'{args.log_dir}/ep{epoch}_student_mel.png')
            save_plot(gt_mel.squeeze().cpu(), f'{args.log_dir}/ep{epoch}_target_mel.png')

            # 6. Save Audio (The Ear Test)
            audio_student = hifigan(pred_rescaled.to('cpu'))
            student_wav_path = f'{args.log_dir}/ep{epoch}_student_audio.wav'
            save_audio(student_wav_path, mel_cfg['sampling_rate'], audio_student)
                
            # 7. Calculate Pitch Tracking Error
            # We use the actual saved wavs so we measure exactly what the vocoder produced
            # If it's not the first epoch, compare to the ep2 ground truth we saved earlier
            f0_mae_val = None

            if os.path.exists(val_target_path):
                f0_mae_val = calculate_f0_error(student_wav_path, val_target_path, sr=mel_cfg['sampling_rate'])
                print(f"--> Epoch {epoch} | F0 Tracking Error (MAE): {f0_mae_val:.2f} Hz\n")

            # -- NEW: Append to Eval CSV --
            with open(eval_log_path, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, mel_mse.item(), f0_mae_val])
