import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import torch
from torch.nn import functional as F
import librosa


def repeat_expand_2d(content, target_len):
    # align content with mel

    src_len = content.shape[-1]
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(content.device)
    temp = torch.arange(src_len+1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos+1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]

    return target


def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()


def save_audio(file_path, sampling_rate, audio):
    audio = np.clip(audio.detach().cpu().squeeze().numpy(), -0.999, 0.999)
    wavfile.write(file_path, sampling_rate, (audio * 32767).astype("int16"))


def minmax_norm_diff(tensor: torch.Tensor, vmax: float = 2.5, vmin: float = -12) -> torch.Tensor:
    tensor = torch.clip(tensor, vmin, vmax)
    tensor = 2 * (tensor - vmin) / (vmax - vmin) - 1
    return tensor


def reverse_minmax_norm_diff(tensor: torch.Tensor, vmax: float = 2.5, vmin: float = -12) -> torch.Tensor:
    tensor = torch.clip(tensor, -1.0, 1.0)
    tensor = (tensor + 1) / 2
    tensor = tensor * (vmax - vmin) + vmin
    return tensor


def calculate_f0_error(student_wav_path, target_wav_path, sr=24000):
    try:
        # Load audio
        y_stud, _ = librosa.load(student_wav_path, sr=sr)
        y_targ, _ = librosa.load(target_wav_path, sr=sr)
        
        # Calculate fundamental frequency using YIN
        # fmin=65 (C2), fmax=1046 (C6) - standard vocal range
        f0_stud = librosa.yin(y_stud, fmin=65, fmax=1046)
        f0_targ = librosa.yin(y_targ, fmin=65, fmax=1046)
        
        # Align lengths in case of minor vocoder padding differences
        min_len = min(len(f0_stud), len(f0_targ))
        
        # Calculate Mean Absolute Error (MAE) in Hz
        f0_error = np.mean(np.abs(f0_stud[:min_len] - f0_targ[:min_len]))
        return f0_error
    except Exception as e:
        return -1.0 # Return -1 if F0 tracking fails