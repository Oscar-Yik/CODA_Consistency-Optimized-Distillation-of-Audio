import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import torch
from torch.nn import functional as F
import librosa
from librosa.core import load
import pyworld as pw


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

# Stolen from another utils
def get_f0(wav_path, wav, method='pyin', padding=True):
    sr = 24000

    if wav is None:
        if wav_path is not None:
            wav, sr = load(wav_path, sr=sr)
        else:
            raise ValueError("Must provide either 'wav' (array) or 'wav_path' (string)")
        
    if method == 'pyin':
        wav = wav[:(wav.shape[0] // 256) * 256]
        wav = np.pad(wav, 384, mode='reflect')
        f0, _, _ = librosa.pyin(wav, frame_length=1024, hop_length=256, center=False, sr=24000,
                                fmin=librosa.note_to_hz('C2'),
                                fmax=librosa.note_to_hz('C6'), fill_na=0)
    elif method == 'world':
        wav = (wav * 32767).astype(np.int16)
        wav = (wav / 32767).astype(np.float64)
        _f0, t = pw.dio(wav, fs=sr, frame_period=256/sr*1000,
                        f0_floor=librosa.note_to_hz('C2'),
                        f0_ceil=librosa.note_to_hz('C6'))
        f0 = pw.stonemask(wav, _f0, t, sr)
        f0 = f0[:-1]

    if padding is True:
        if f0.shape[-1] % 8 !=0:
            f0 = np.pad(f0, ((0, 8-f0.shape[-1] % 8)), 'constant', constant_values=0)

    return f0

def calculate_audio_mse(original_wav_path, streamed_wav_path):
    sr = 24000
    
    wav_orig, _ = librosa.load(original_wav_path, sr=sr)
    wav_stream, _ = librosa.load(streamed_wav_path, sr=sr)

    min_len = min(len(wav_orig), len(wav_stream))
    wav_orig = wav_orig[:min_len]
    wav_stream = wav_stream[:min_len]

    mel_orig = librosa.feature.melspectrogram(y=wav_orig, sr=sr, n_fft=1024, hop_length=256, n_mels=100)
    mel_stream = librosa.feature.melspectrogram(y=wav_stream, sr=sr, n_fft=1024, hop_length=256, n_mels=100)

    log_mel_orig = librosa.power_to_db(mel_orig, ref=np.max)
    log_mel_stream = librosa.power_to_db(mel_stream, ref=np.max)

    tensor_orig = torch.from_numpy(log_mel_orig)
    tensor_stream = torch.from_numpy(log_mel_stream)

    mse = F.mse_loss(tensor_orig, tensor_stream).item()

    return mse