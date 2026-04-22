import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn

from pitch_controller.modules.BigVGAN.inference import load_model


PRECISION_TO_DTYPE = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}


def resolve_dtype(precision):
    try:
        return PRECISION_TO_DTYPE[precision]
    except KeyError:
        raise ValueError(f"Unknown precision: {precision!r}. Expected one of {list(PRECISION_TO_DTYPE)}.")


class GriffinLimVocoder:
    def __init__(self, sr=24000, n_fft=1024, hop_length=256, n_mels=100, f_max=12000, n_iter=32, device='cpu'):
        mel_basis = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=f_max)
        self.mel_basis_pinv = torch.from_numpy(np.linalg.pinv(mel_basis)).float().to(device)
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, n_iter=n_iter,
        ).to(device)

    def __call__(self, mel_spectrogram):
        mel = mel_spectrogram.squeeze(0).to(self.mel_basis_pinv.dtype)
        linear = torch.clamp(self.mel_basis_pinv @ torch.exp(mel), min=0)
        return self.griffin_lim(linear).unsqueeze(0)


class BigVGANVocoder:
    def __init__(self, checkpoint_path, device='cpu', compile_model=False, compile_backend='inductor'):
        self.model, _ = load_model(checkpoint_path, device=device)
        self.model.eval()
        if compile_model:
            self.model = torch.compile(self.model, backend=compile_backend)

    def __call__(self, mel_spectrogram):
        return self.model(mel_spectrogram).float()


class VocosVocoder:
    def __init__(self, model_name='charactr/vocos-mel-24khz', sr=24000, n_fft=1024, n_mels=100,
                 f_max=12000, device='cpu', compile_model=False, compile_backend='inductor'):
        from vocos import Vocos
        self.model = Vocos.from_pretrained(model_name).to(device)
        self.model.eval()

        # Project mels use librosa (Slaney norm, fmax=12000). Vocos was trained on
        # torchaudio's HTK mel with fmax=None. Precompute a basis to convert between them.
        src_basis = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=f_max)
        tgt_basis = self.model.feature_extractor.mel_spec.mel_scale.fb.cpu().numpy().T
        convert = tgt_basis @ np.linalg.pinv(src_basis)
        self.convert = torch.from_numpy(convert).float().to(device)

        if compile_model:
            self.model.backbone = torch.compile(self.model.backbone, backend=compile_backend)

    def __call__(self, mel_spectrogram):
        mel = mel_spectrogram.squeeze(0).float()
        linear = torch.clamp(torch.exp(mel), min=0)
        vocos_mel = torch.log(torch.clamp(self.convert @ linear, min=1e-5)).unsqueeze(0)
        return self.model.decode(vocos_mel).float()


def load_vocoder(config, mel_cfg, device):
    vocoder_type = config['models'].get('vocoder_type', 'vocos')
    perf = config.get('performance', {})
    compile_model = perf.get('compile', False)
    compile_backend = perf.get('compile_backend', 'inductor')
    precision = perf.get('precision', 'fp32')

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
    elif vocoder_type == 'vocos':
        vocoder = VocosVocoder(
            model_name=config['models'].get('vocos_model', 'charactr/vocos-mel-24khz'),
            sr=mel_cfg['sampling_rate'],
            n_fft=mel_cfg['n_fft'],
            n_mels=mel_cfg['n_mels'],
            device=device,
            compile_model=compile_model,
            compile_backend=compile_backend,
        )
        print(f"Using Vocos vocoder on device: {device} (precision={precision} via autocast, compile={compile_model}, backend={compile_backend})")
    elif vocoder_type == 'bigvgan':
        vocoder = BigVGANVocoder(
            checkpoint_path=config['models']['vocoder_checkpoint'],
            device=device,
            compile_model=compile_model,
            compile_backend=compile_backend,
        )
        print(f"Using BigVGAN vocoder on device: {device} (precision={precision} via autocast, compile={compile_model}, backend={compile_backend})")
    else:
        raise ValueError(f"Unknown vocoder_type: {vocoder_type!r}. Expected 'griffin_lim', 'vocos', or 'bigvgan'.")
    return vocoder, vocoder_type
