import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn

from pitch_controller.modules.BigVGAN.inference import load_model


class GriffinLimVocoder:
    def __init__(self, sr=24000, n_fft=1024, hop_length=256, n_mels=100, f_max=12000, n_iter=32, device='cpu'):
        mel_basis = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=f_max)
        self.mel_basis_pinv = torch.from_numpy(np.linalg.pinv(mel_basis)).float().to(device)
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, n_iter=n_iter,
        ).to(device)

    def __call__(self, mel_spectrogram):
        linear = torch.clamp(self.mel_basis_pinv @ torch.exp(mel_spectrogram.squeeze(0)), min=0)
        return self.griffin_lim(linear).unsqueeze(0)


def load_vocoder(config, mel_cfg, device):
    vocoder_type = config['models'].get('vocoder_type', 'bigvgan')
    if vocoder_type == 'griffin_lim':
        vocoder = GriffinLimVocoder(
            sr=mel_cfg['sampling_rate'],
            n_fft=mel_cfg['n_fft'],
            hop_length=mel_cfg['hop_size'],
            n_mels=mel_cfg['n_mels'],
            n_iter=32,
            device=device,
        )
        print("Using Griffin-Lim vocoder (GPU)" if device == 'cuda' else "Using Griffin-Lim vocoder (CPU)")
    else:
        vocoder, _ = load_model(config['models']['vocoder_checkpoint'], device=device)
        vocoder.eval()
        print(f"Vocoder on device: {next(vocoder.parameters()).device}")
    return vocoder, vocoder_type
