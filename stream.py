import librosa
import torch
import yaml
import json

from pitch_controller.models.consistency import ConsistencyPitcher
from pitch_controller.models.unet import UNetPitcher
from pitch_controller.utils import minmax_norm_diff, reverse_minmax_norm_diff
from streaming.audio_streamer import AudioStreamer
from streaming.vocoder import load_vocoder
from utils import get_matched_f0, get_world_mel, log_f0
from diffusers import DDIMScheduler
import numpy as np


def load_config(config_path='streaming/config.json'):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_audio_processor(model, hifigan, noise_scheduler, device, config):
    """
    Factory function that creates an audio processing callback.
    This closure captures the model and configuration.
    """
    min_mel = config['processing']['min_mel']
    max_mel = config['processing']['max_mel']
    sr = config['audio']['target_sample_rate']
    pitch_config = config['pitch']

    @torch.no_grad()
    def process_audio(audio_window):
        """
        Process audio window and return modified audio.

        Args:
            audio_window: numpy array of audio samples (at target sample rate)

        Returns:
            numpy array of processed audio (at raw sample rate for playback)
        """
        import time
        total_start = time.perf_counter()

        # Preprocessing
        preprocess_start = time.perf_counter()
        source_mel = get_world_mel(wav_path=None, sr=sr, wav=audio_window)
        f0_ref = get_matched_f0(x_wav=audio_window, y_wav=audio_window, method='world')
        f0_ref = log_f0(f0_ref, {
            'f0_bin': pitch_config['f0_bin'],
            'f0_min': librosa.note_to_hz(pitch_config['f0_min_note']),
            'f0_max': librosa.note_to_hz(pitch_config['f0_max_note'])
        })
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000

        source_mel = torch.from_numpy(source_mel).float().unsqueeze(0).to(device)
        f0_ref = torch.from_numpy(f0_ref).float().unsqueeze(0).to(device)
        source_x = minmax_norm_diff(source_mel, vmax=max_mel, vmin=min_mel)

        # Model inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_start = time.perf_counter()
        t = torch.tensor([0], device=device)
        model_output = model(x=source_x, t=t, mean=source_x, f0=f0_ref, noise_scheduler=noise_scheduler)
        pred_mel = reverse_minmax_norm_diff(model_output, vmax=max_mel, vmin=min_mel)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = (time.perf_counter() - model_start) * 1000

        # Vocoder
        vocoder_start = time.perf_counter()
        output_wav = hifigan(pred_mel)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        output_wav = output_wav.squeeze().cpu().numpy().astype(np.float32)
        vocoder_time = (time.perf_counter() - vocoder_start) * 1000

        total_time = (time.perf_counter() - total_start) * 1000
        print(f"Total: {total_time:.1f}ms | Preprocess: {preprocess_time:.1f}ms | Model: {model_time:.1f}ms | Vocoder: {vocoder_time:.1f}ms")

        return output_wav

    return process_audio


if __name__ == '__main__':
    config = load_config()
    use_gpu = torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'

    print(f"Using device: {device}")
    if use_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model configuration
    model_config = yaml.load(open(config['models']['diffusion_config']), Loader=yaml.FullLoader)
    mel_cfg = model_config['logmel']
    ddpm_cfg = model_config['ddpm']
    unet_cfg = model_config['unet']

    # Initialize noise scheduler (only need alphas_cumprod for single-step inference)
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=ddpm_cfg['num_train_steps'],
        beta_schedule='squaredcos_cap_v2'
    )
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    # Load consistency model
    unet = UNetPitcher(**unet_cfg)
    model = ConsistencyPitcher(unet, sigma_data=0.5).to(device)

    state_dict = torch.load(config['models']['consistency_checkpoint'], map_location=device, weights_only=True)
    for key in list(state_dict.keys()):
        state_dict[key.replace('_orig_mod.', '')] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model on device: {next(model.parameters()).device}")

    # Load vocoder
    hifigan, vocoder_type = load_vocoder(config, mel_cfg, device)

    print("Models loaded successfully")

    # Warmup pass to avoid cold start
    print("Warming up models...")
    import time
    dummy_mel = torch.randn(1, mel_cfg['n_mels'], 8).to(device)
    dummy_f0 = torch.randn(1, 8).to(device)
    t = torch.tensor([0], device=device)

    warmup_start = time.perf_counter()
    with torch.no_grad():
        _ = model(x=dummy_mel, t=t, mean=dummy_mel, f0=dummy_f0, noise_scheduler=noise_scheduler)
        _ = hifigan(dummy_mel)
    if use_gpu:
        torch.cuda.synchronize()
    warmup_time = (time.perf_counter() - warmup_start) * 1000
    print(f"Warmup complete: {warmup_time:.1f}ms")

    # Streamer
    audio_processor = create_audio_processor(model, hifigan, noise_scheduler, device, config)
    streamer = AudioStreamer(audio_callback=audio_processor)
    streamer.run()
