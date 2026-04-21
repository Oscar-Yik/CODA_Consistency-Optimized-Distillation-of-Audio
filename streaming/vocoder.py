import librosa
import numpy as np
import torch

from pitch_controller.modules.BigVGAN.inference import load_model


class GriffinLimVocoder:
    def __init__(self, sr=24000, n_fft=1024, hop_length=256, n_iter=32):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_iter = n_iter

    def __call__(self, mel_spectrogram):
        mel = mel_spectrogram.squeeze(0).cpu().numpy()
        mel_linear = np.exp(mel)
        wav = librosa.feature.inverse.mel_to_audio(
            mel_linear,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_iter=self.n_iter
        )
        return torch.from_numpy(wav).unsqueeze(0)


def load_vocoder(config, mel_cfg, device):
    vocoder_type = config['models'].get('vocoder_type', 'bigvgan')
    if vocoder_type == 'griffin_lim':
        vocoder = GriffinLimVocoder(
            sr=mel_cfg['sampling_rate'],
            n_fft=mel_cfg['n_fft'],
            hop_length=mel_cfg['hop_size'],
        )
        print("Using Griffin-Lim vocoder")
    else:
        vocoder, _ = load_model(config['models']['vocoder_checkpoint'], device=device)
        vocoder.eval()
        print(f"Vocoder on device: {next(vocoder.parameters()).device}")
    return vocoder, vocoder_type
