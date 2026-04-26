import yaml
import json
from tqdm import tqdm
import numpy as np
from types import SimpleNamespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler

from dataset import VCDecLPCDataset, VCDecLPCBatchCollate
from models.unet import UNetPitcher
from models.consistency import ConsistencyPitcher
from modules.BigVGAN.inference import load_model
from utils import save_audio, minmax_norm_diff, reverse_minmax_norm_diff, get_f0, calculate_audio_mse

import matplotlib.pyplot as plt
import torch.nn.functional as F

import seaborn as sns
import pandas as pd

GRAPH_DIR = Path("graphs")

def load(args, config, device):
    mel_cfg = config['logmel']
    ddpm_cfg = config['ddpm']
    unet_cfg = config['unet']

    train_set = VCDecLPCDataset(args.data_dir, subset='test', content_dir=args.lpc_dir, f0_type="log")
    collate_fn = VCDecLPCBatchCollate(args.train_frames)
    loader = DataLoader(train_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))

    gt_mel = batch['mel1'].to(device)
    content = batch['content1'].to(device)
    f0 = batch['f0_1'].to(device).long()
    content_norm = minmax_norm_diff(content, vmax=mel_cfg['max'], vmin=mel_cfg['min'])

    noise_scheduler = DDIMScheduler(num_train_timesteps=ddpm_cfg['num_train_steps'])
    noise_scheduler.set_timesteps(ddpm_cfg['inference_steps'])
    generator = torch.Generator(device=device).manual_seed(2024)

    print("Loading Teacher...")
    teacher = UNetPitcher(**unet_cfg).to(device)
    t_state = torch.load('../ckpts/world_fixed_40.pt', map_location=device, weights_only=True)
    for k in list(t_state.keys()): t_state[k.replace('_orig_mod.', '')] = t_state.pop(k)
    teacher.load_state_dict(t_state)
    teacher.eval()

    print("Loading Student...")
    student_unet = UNetPitcher(**unet_cfg).to(device)
    student = ConsistencyPitcher(student_unet).to(device)
    s_state = torch.load(args.ckpt, map_location=device, weights_only=True)
    student.load_state_dict(s_state)
    student.eval()

    return student, teacher, generator, noise_scheduler, gt_mel, f0, content_norm, mel_cfg

def teacher_mel(teacher, generator, noise_scheduler, f0, content_norm, mel_cfg, device, target_step_idx):
    teacher_pred = teacher_inference(teacher, generator, noise_scheduler, f0, content_norm, mel_cfg, device, eta=1, target_step_idx=target_step_idx)
    teacher_mel_show = teacher_pred.squeeze().cpu().numpy()
    return teacher_mel_show

def teacher_mse(teacher, generator, noise_scheduler, gt_mel, f0, content_norm, mel_cfg, device):
    teacher_pred = teacher_inference(teacher, generator, noise_scheduler, f0, content_norm, mel_cfg, device)
    teacher_100_mse = F.mse_loss(teacher_pred, gt_mel).item()
    print(f"Teacher Baseline (100 steps) MSE: {teacher_100_mse:.4f}")
    return teacher_100_mse

def teacher_inference(teacher, generator, noise_scheduler, f0, content_norm, mel_cfg, device, eta=1, target_step_idx=None):
    print("Running Teacher 100-Step Baseline...")
    x_t_teacher = torch.randn_like(content_norm, generator=generator).to(device)
    with torch.no_grad():
        for i, t in enumerate(tqdm(noise_scheduler.timesteps, desc="Teacher")):
            t_batch = torch.tensor([t.item()] * x_t_teacher.shape[0], device=device)
            model_output = teacher(x=x_t_teacher, mean=content_norm, f0=f0, t=t_batch)

            step_output = noise_scheduler.step(model_output, t.item(), x_t_teacher, eta=eta, generator=generator)
            x_t_teacher = step_output.prev_sample

            if target_step_idx is not None and i == target_step_idx:
                current_guess = step_output.pred_original_sample
                return reverse_minmax_norm_diff(current_guess, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
            
    teacher_pred = reverse_minmax_norm_diff(x_t_teacher, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
    return teacher_pred

def student_mel(student, generator, noise_scheduler, f0, content_norm, mel_cfg, device):
    milestones = [[0, 25, 50, 75]]
    student_preds = student_inference(student, generator, noise_scheduler, f0, content_norm, mel_cfg, device, milestones)
    student_mel_show = student_preds[0].squeeze().cpu().numpy()
    return student_mel_show

def student_mse(student, generator, noise_scheduler, gt_mel, f0, content_norm, mel_cfg, device):
    milestones = [[0], [0, 50], [0, 30, 60], [0, 25, 50, 75]]
    student_preds = student_inference(student, generator, noise_scheduler, f0, content_norm, mel_cfg, device, milestones)
    student_mses = [F.mse_loss(student_pred, gt_mel).item() for student_pred in student_preds]
    return student_mses

def student_inference(student, generator, noise_scheduler, f0, content_norm, mel_cfg, device, milestones):
    print("Evaluating Student at key milestones...")
    student_preds = []

    with torch.no_grad():
        for chain in milestones:
            current_input = torch.randn_like(content_norm, generator=generator).to(device)
            
            for step_num, current_t_idx in enumerate(chain):
                t_current = torch.as_tensor([noise_scheduler.timesteps[current_t_idx]], device=device)
                pred_x0 = student(current_input, t_current, content_norm, f0, noise_scheduler)
                
                if step_num < len(chain) - 1:
                    next_t_idx = chain[step_num + 1]
                    t_next = torch.as_tensor([noise_scheduler.timesteps[next_t_idx]], device=device)
                    fresh_noise = torch.randn_like(pred_x0, generator=generator)
                    current_input = noise_scheduler.add_noise(pred_x0, fresh_noise, t_next)
                else:
                    final_pred = pred_x0
                    
            s_pred_rescaled = reverse_minmax_norm_diff(final_pred, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
            student_preds.append(s_pred_rescaled)

    return student_preds

def generate_bar_graph(student_mses, teacher_100_mse):
    print("Generating Seaborn Graph...")
    
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 6))
    
    step_counts = range(1, len(student_mses) + 1)
    labels = [f'{s} Step{"s" if s > 1 else ""}' for s in step_counts]
    all_labels = labels + ['Teacher\n(100 Steps)']
    all_errors = student_mses + [teacher_100_mse]

    categories = ['Consistency Student'] * len(student_mses) + ['Original Teacher']

    df = pd.DataFrame({
        'Model Configuration': all_labels,
        'Mean Squared Error (MSE)': all_errors,
        'Model Type': categories
    })

    nice_green = sns.color_palette("muted")[2] 
    nice_red = sns.color_palette("muted")[3]

    ax = sns.barplot(
        data=df, 
        x='Model Configuration', 
        y='Mean Squared Error (MSE)', 
        hue='Model Type',
        palette={'Consistency Student': nice_green, 'Original Teacher': nice_red},
        dodge=False,
        edgecolor='black',
        linewidth=1.5
    )

    ax.set_yscale("log")

    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=5, fontweight='bold', size=11)

    plt.axhline(y=teacher_100_mse, color='black', linestyle='--', alpha=0.6, label='Teacher Target Quality')

    plt.title('Inference Efficiency: Consistency vs. Baseline', fontweight='bold', pad=20)
    plt.ylabel('Mean Squared Error (MSE) [Log Scale]', fontweight='bold')
    plt.xlabel('') # Remove redundant X-axis label
    
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, loc='upper right', framealpha=0.9)
    sns.despine()

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'bar_chart_efficiency_seaborn.pdf')
    print("Saved clean Seaborn bar chart to 'bar_chart_efficiency_seaborn.pdf'!")

def generate_spectrograms(gt_mel_show, student_mel_show, teacher_mel_show):
    print("Generating Side-by-Side Spectrograms...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmin = min(gt_mel_show.min(), student_mel_show.min(), teacher_mel_show.min())
    vmax = max(gt_mel_show.max(), student_mel_show.max(), teacher_mel_show.max())
    
    im0 = axes[0].imshow(gt_mel_show, aspect="auto", origin="lower", interpolation='none', vmin=vmin, vmax=vmax)
    axes[0].set_title('Ground Truth (Original Audio)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time Frames')
    axes[0].set_ylabel('Mel Frequency Bins')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(teacher_mel_show, aspect="auto", origin="lower", interpolation='none', vmin=vmin, vmax=vmax)
    axes[1].set_title('Original UNet (4 Steps DDIM)', fontsize=14, fontweight='bold', color='#d62728')
    axes[1].set_xlabel('Time Frames')
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(student_mel_show, aspect="auto", origin="lower", interpolation='none', vmin=vmin, vmax=vmax)
    axes[2].set_title('Consistency Model (4 Chain Steps)', fontsize=14, fontweight='bold', color='#2ca02c')
    axes[2].set_xlabel('Time Frames')
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'spectrogram_comparison_4steps.pdf')
    print("Saved comparison to 'spectrogram_comparison_4steps.pdf'!")

def plot_side_by_side_mel_spectrogram(args, config, device):

    student, teacher, generator, noise_scheduler, gt_mel, f0, content_norm, mel_cfg = load(args, config, device)

    gt_mel_show = gt_mel.squeeze().cpu().numpy()
    student_mel_show = student_mel(student, generator, noise_scheduler, f0, content_norm, mel_cfg, device)
    target_step_idx = 3
    teacher_mel_show = teacher_mel(teacher, generator, noise_scheduler, f0, content_norm, mel_cfg, device, target_step_idx)

    generate_spectrograms(gt_mel_show, student_mel_show, teacher_mel_show)

def plot_efficiency_barchart(args, config, device, num_samples=100):

    student, teacher, generator, noise_scheduler, _, _, _, mel_cfg = load(args, config, device)

    train_set = VCDecLPCDataset(args.data_dir, subset='test', content_dir=args.lpc_dir, f0_type="log")
    collate_fn = VCDecLPCBatchCollate(args.train_frames)
    loader = DataLoader(train_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    milestones = [[0], [0, 50], [0, 30, 60], [0, 25, 50, 75]] 
    
    teacher_mse_sum = 0.0
    student_mse_sums = [0.0] * len(milestones)
    
    actual_samples = min(num_samples, len(loader))
    
    print(f"\n--- Starting Robust Evaluation over {actual_samples} Samples ---")
    
    for i, batch in enumerate(loader):
        if i >= actual_samples:
            break
            
        print(f"\nEvaluating Sample {i+1}/{actual_samples}...")
        
        gt_mel = batch['mel1'].to(device)
        content = batch['content1'].to(device)
        f0 = batch['f0_1'].to(device).long()
        content_norm = minmax_norm_diff(content, vmax=mel_cfg['max'], vmin=mel_cfg['min'])
        
        t_mse = teacher_mse(teacher, generator, noise_scheduler, gt_mel, f0, content_norm, mel_cfg, device)
        s_mses = student_mse(student, generator, noise_scheduler, gt_mel, f0, content_norm, mel_cfg, device)
        
        teacher_mse_sum += t_mse
        for j in range(len(milestones)):
            student_mse_sums[j] += s_mses[j]

    avg_teacher_mse = teacher_mse_sum / actual_samples
    avg_student_mses = [s_sum / actual_samples for s_sum in student_mse_sums]

    print(f"\n--- Final Averages ({actual_samples} Samples) ---")
    print(f"Teacher Baseline (100 steps) MSE: {avg_teacher_mse:.4f}")
    for j, chain in enumerate(milestones):
        print(f"Student ({len(chain)} steps) MSE: {avg_student_mses[j]:.4f}")

    generate_bar_graph(avg_student_mses, avg_teacher_mse)

def generate_and_save_audio(args, config, device):

    student, teacher, generator, noise_scheduler, gt_mel, f0, content_norm, mel_cfg = load(args, config, device)
    hifigan, _ = load_model(args.vocoder_dir, device=device) 
    milestones = [[0, 25, 50, 75]]

    teacher_pred = teacher_inference(teacher, generator, noise_scheduler, f0, content_norm, mel_cfg, device)
    student_preds = student_inference(student, generator, noise_scheduler, f0, content_norm, mel_cfg, device, milestones)

    audio_student = hifigan(student_preds[0])
    save_audio(GRAPH_DIR / 'student_4_step.wav', mel_cfg['sampling_rate'], audio_student)

    audio_teacher = hifigan(teacher_pred)
    save_audio(GRAPH_DIR / 'teacher.wav', mel_cfg['sampling_rate'], audio_teacher)

    audio_target = hifigan(gt_mel.to(device))
    save_audio(GRAPH_DIR / 'ground_truth.wav', mel_cfg['sampling_rate'], audio_target)

def plot_f0_comparison():

    wav_path1 = GRAPH_DIR / "autotuned_emma_twinkle.wav"
    wav_path2 = GRAPH_DIR / "emma_twinkle.wav"
    output_path = GRAPH_DIR / "f0_comparison_seaborn.pdf"
    label1 = "Autotuned"
    label2 = "Original"
    sr = 24000
    hop_length = 256

    f0_1 = get_f0(wav_path1, wav=None, method='world', padding=False)
    f0_2 = get_f0(wav_path2, wav=None, method='world', padding=False)
    
    f0_1_clean = np.where(f0_1 > 0, f0_1, np.nan)
    f0_2_clean = np.where(f0_2 > 0, f0_2, np.nan)
    
    time1 = np.arange(len(f0_1)) * (hop_length / sr)
    time2 = np.arange(len(f0_2)) * (hop_length / sr)
    
    df1 = pd.DataFrame({'Time (seconds)': time1, 'Frequency (Hz)': f0_1_clean, 'Version': label1})
    df2 = pd.DataFrame({'Time (seconds)': time2, 'Frequency (Hz)': f0_2_clean, 'Version': label2})
    df = pd.concat([df1, df2], ignore_index=True)
    
    print("Generating Seaborn F0 Graph...")
    
    sns.set_theme(style="darkgrid", context="talk")
    plt.figure(figsize=(12, 6))
    
    custom_palette = sns.color_palette("husl", 2)
    
    ax = sns.lineplot(
        data=df, 
        x='Time (seconds)', 
        y='Frequency (Hz)', 
        hue='Version',
        style='Version',
        palette=custom_palette,
        linewidth=2.5,
        alpha=0.9
    )

    # grid_color = sns.axes_style()["grid.color"]
    palette = sns.color_palette("muted", 3)

    # Notes
    for (y, label), color in zip([(233.08, "B Flat"), (349.23, "F"), (392, "G")], palette):
        ax.axhline(y=y, color=color, linestyle="--", linewidth=1.2, alpha=0.7, zorder=0)
        ax.text(1.01, y, f"{label}", color=color, 
                va='center', ha='left', 
                transform=ax.get_yaxis_transform(), 
                fontsize=11, fontweight='bold', alpha=0.9)
    
    plt.title(f"F0 Pitch Contour: {label1} vs {label2}", fontweight='bold', pad=15)
    
    combined_f0 = np.concatenate([f0_1, f0_2])
    voiced_points = combined_f0[combined_f0 > 0]
    
    if len(voiced_points) > 0:
        plt.ylim(voiced_points.min() - 15, voiced_points.max() + 15)
    
    plt.legend(title='', loc='upper right', framealpha=0.9)
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    print(f"Saved F0 comparison to '{output_path}'!")

def plot_streaming_tradeoffs(calculate_mse = False):
    print("Generating Stacked Trade-off Graphs...")
    
    sizes = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    mses = [223.527, 166.592, 158.925, 153.565, 76.477, 68.271, 58.853, 56.268, 49.164]
    if calculate_mse:
        original = GRAPH_DIR / "emma_twinkle.wav"
        sizes = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
        streamed = [GRAPH_DIR / f"streamed_output_{size}.wav" for size in sizes]
        mses = []
        for stream_audio in streamed:
            mse = calculate_audio_mse(original, stream_audio)
            mses.append(mse)

    window_chunks = [str(round(chunk/24)) for chunk in sizes]

    data = {
        'Window Size (ms)': window_chunks,
        'Inference Latency (ms)': [37.322, 35.118, 35.082, 35.376, 36.744, 42.325, 41.43, 57.54, 56.441],
        'MSE Error': mses
    }
    df = pd.DataFrame(data)

    sns.set_theme(style="darkgrid", context="talk")
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    nice_red = sns.color_palette("muted")[3]
    nice_green = sns.color_palette("muted")[2]

    sns.lineplot(data=df, x='Window Size (ms)', y='Inference Latency (ms)', ax=ax1, 
                 color=nice_red, marker='o', markersize=10, linewidth=3)
    
    ax1.set_title("The Streaming Trade-off", fontweight='bold', pad=20, fontsize=18)
    ax1.set_ylabel("Processing Latency\n(Milliseconds)", fontweight='bold')

    sns.lineplot(data=df, x='Window Size (ms)', y='MSE Error', ax=ax2, 
                 color=nice_green, marker='s', markersize=10, linewidth=3)
    
    ax2.set_ylabel("Audio Distortion\n(Mean Squared Error)", fontweight='bold')
    ax2.set_xlabel("Audio Window Size (ms)", fontweight='bold')

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    plt.savefig(GRAPH_DIR / 'streaming_tradeoffs_stacked.pdf', dpi=300)
    print("Saved dual graphs to 'streaming_tradeoffs_stacked.pdf'!")

if __name__ == "__main__":

    # Setup
    with open("test_config.json", "r") as f:
        args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    GRAPH_DIR.mkdir(exist_ok=True)

    # Produce all graphs
    plot_efficiency_barchart(args, config, device)
    # generate_bar_graph([19.0729, 0.8148, 0.3251, 0.2519], 0.2840)
    plot_side_by_side_mel_spectrogram(args, config, device)
    generate_and_save_audio(args, config, device)
    plot_f0_comparison()
    plot_streaming_tradeoffs(calculate_mse=False)
