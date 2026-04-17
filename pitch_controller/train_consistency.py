import os, argparse, yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from diffusers import DDIMScheduler

from dataset import VCDecLPCDataset, VCDecLPCBatchCollate, VCDecLPCTest
from models.unet import UNetPitcher
from modules.BigVGAN.inference import load_model
from cd_trainer import ConsistencyTrainer
from utils import save_plot, save_audio
from utils import minmax_norm_diff, reverse_minmax_norm_diff


parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, default='config/DiffWorld_24k.yaml')

parser.add_argument('-seed', type=int, default=98)
parser.add_argument('-amp', type=bool, default=True)
parser.add_argument('-compile', type=bool, default=False)

parser.add_argument('-data_dir', type=str, default='../data/')
parser.add_argument('-lpc_dir', type=str, default='world')
parser.add_argument('-vocoder_dir', type=str, default='../ckpts/bigvgan_24khz_100band/g_05000000.pt')

parser.add_argument('-train_frames', type=int, default=128)
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-test_size', type=int, default=1)
parser.add_argument('-num_workers', type=int, default=4)
parser.add_argument('-lr', type=float, default=5e-5)
parser.add_argument('-weight_decay', type=int, default=1e-6)

parser.add_argument('-epochs', type=int, default=80)
parser.add_argument('-save_every', type=int, default=2)
parser.add_argument('-log_step', type=int, default=100)
parser.add_argument('-log_dir', type=str, default='logs_consistency')
parser.add_argument('-ckpt_dir', type=str, default='ckpt_consistency')

args = parser.parse_args()
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
    collate_fn = VCDecLPCBatchCollate(args.train_frames)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True)

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

    # prepare DPM scheduler
    noise_scheduler = DDIMScheduler(num_train_timesteps=ddpm_cfg['num_train_steps'])
    noise_scheduler.set_timesteps(ddpm_cfg['inference_steps']) 

    print('Start Consistency Distillation.')
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch: {epoch} [iteration: {global_step}]')
        trainer.student.train()
        losses = []

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

            losses.append(loss)
            global_step += 1

            # logging
            if global_step % args.log_step == 0:
                losses = np.asarray(losses)
                msg = '\nEpoch: [{}][{}]\t' \
                        'Batch: [{}][{}]\tLoss: {:.6f}\n'.format(epoch,
                                                                args.epochs,
                                                                step+1,
                                                                len(train_loader),
                                                                np.mean(losses))
                with open(f'{args.log_dir}/train.log', 'a') as f:
                    f.write(msg)
                losses = []

        if epoch % args.save_every > 0:
            continue

        print('Saving model...\n')
        torch.save(trainer.student.state_dict(), f"{args.ckpt_dir}/consistency_model_{epoch}.pt")

        print('Inference...\n')
        noise = None
        noise_scheduler.set_timesteps(ddpm_cfg['inference_steps'])
        trainer.student.eval()
        with torch.no_grad():
            # Take the first mel from the batch as a test
            test_mean = mean[0:1]  # Use content representation, not target mel
            test_f0 = f0[0:1]

            # use pure noise
            noise = torch.randn_like(test_mean)
            t_max = noise_scheduler.timesteps[0]

            pred = trainer.student(noise, t_max, test_mean, test_f0, noise_scheduler)
            
            # Save audio sample
            pred_rescaled = reverse_minmax_norm_diff(pred, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
            audio = hifigan(pred_rescaled.to('cpu'))
            save_audio(f'{args.log_dir}/sample_ep{epoch}.wav', mel_cfg['sampling_rate'], audio)

        args.save_ori = False
