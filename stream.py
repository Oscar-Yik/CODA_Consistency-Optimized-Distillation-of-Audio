import librosa
import torch
import yaml
import json

from pitch_controller.models.consistency import ConsistencyPitcher
from pitch_controller.models.unet import UNetPitcher
from pitch_controller.utils import minmax_norm_diff, reverse_minmax_norm_diff
from streaming.audio_streamer import AudioStreamer
from streaming.vocoder import load_vocoder, resolve_dtype
from utils import get_matched_f0, get_world_mel, log_f0
from diffusers import DDIMScheduler
import numpy as np


def load_config(config_path='streaming/config.json'):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_audio_processor(model, hifigan, noise_scheduler, device, config, chain_timesteps):
    """
    Factory function that creates an audio processing callback.
    This closure captures the model and configuration.

    chain_timesteps: 1-D tensor of real scheduler timesteps (high -> low), one
        per consistency iteration. Sampling starts from pure noise, predicts
        x0 at chain_timesteps[0], then re-noises to chain_timesteps[1] and
        repeats. Mirrors the chain used in test_consistency.py.
    """
    min_mel = config['processing']['min_mel']
    max_mel = config['processing']['max_mel']
    sr = config['audio']['target_sample_rate']
    pitch_config = config['pitch']
    dtype = resolve_dtype(config.get('performance', {}).get('precision', 'fp32'))
    use_autocast = dtype != torch.float32
    autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

        # Model inference: start from pure noise, chain through real timesteps
        # (mirrors test_consistency.py's sampling loop).
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_start = time.perf_counter()

        with torch.autocast(device_type=autocast_device, dtype=dtype, enabled=use_autocast):
            current_input = torch.randn_like(source_x)
            num_steps = len(chain_timesteps)
            for step_num in range(num_steps):
                t_current = chain_timesteps[step_num].view(1)
                pred_x0 = model(x=current_input, t=t_current, mean=source_x, f0=f0_ref, noise_scheduler=noise_scheduler)
                if step_num < num_steps - 1:
                    t_next = chain_timesteps[step_num + 1].view(1)
                    fresh_noise = torch.randn_like(pred_x0)
                    current_input = noise_scheduler.add_noise(pred_x0, fresh_noise, t_next)

        pred_mel = reverse_minmax_norm_diff(pred_x0.float(), vmax=max_mel, vmin=min_mel)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = (time.perf_counter() - model_start) * 1000

        # Vocoder
        vocoder_start = time.perf_counter()
        with torch.autocast(device_type=autocast_device, dtype=dtype, enabled=use_autocast):
            output_wav = hifigan(pred_mel)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        output_wav = output_wav.squeeze().cpu().numpy().astype(np.float32)
        vocoder_time = (time.perf_counter() - vocoder_start) * 1000

        total_time = (time.perf_counter() - total_start) * 1000
        print(f"Total: {total_time:.1f}ms | Preprocess: {preprocess_time:.1f}ms | Model: {model_time:.1f}ms | Vocoder: {vocoder_time:.1f}ms")

        return output_wav, {
            'preprocess': preprocess_time,
            'model': model_time,
            'vocoder': vocoder_time,
        }

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

    # Initialize noise scheduler. Match test_consistency.py: use inference_steps
    # from the diffusion config so chain indices [0, 30, 60] map to the same
    # real timesteps the student was evaluated on.
    noise_scheduler = DDIMScheduler(num_train_timesteps=ddpm_cfg['num_train_steps'])
    noise_scheduler.set_timesteps(ddpm_cfg['inference_steps'])
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    # Load consistency model
    perf_cfg = config.get('performance', {})
    precision = perf_cfg.get('precision', 'fp32')
    compile_model = perf_cfg.get('compile', False)
    compile_backend = perf_cfg.get('compile_backend', 'inductor')
    dtype = resolve_dtype(precision)

    # Build the sampling chain: real scheduler timesteps ordered high -> low.
    # `performance.chain_indices` in config, when present, is used directly.
    # Otherwise fall back to [0, 30, 60] (matching test_consistency.py) when
    # the schedule is long enough, else evenly spaced across the schedule for
    # `performance.consistency_iterations` (default 3) steps.
    total_steps = len(noise_scheduler.timesteps)
    configured_indices = perf_cfg.get('chain_indices')
    if configured_indices is not None:
        chain_indices = [int(i) for i in configured_indices]
        if not chain_indices:
            raise ValueError("performance.chain_indices must be non-empty")
        if any(i < 0 or i >= total_steps for i in chain_indices):
            raise ValueError(f"performance.chain_indices must all be in [0, {total_steps}) (got {chain_indices})")
    else:
        consistency_iterations = int(perf_cfg.get('consistency_iterations', 3))
        if consistency_iterations < 1:
            raise ValueError(f"performance.consistency_iterations must be >= 1 (got {consistency_iterations})")
        else:
            chain_indices = [int(round(i * (total_steps - 1) / max(consistency_iterations - 1, 1)))
                             for i in range(consistency_iterations)]
    chain_timesteps = torch.as_tensor(
        [noise_scheduler.timesteps[i].item() for i in chain_indices],
        dtype=torch.long, device=device,
    )
    print(f"Consistency chain indices: {chain_indices} (timesteps={chain_timesteps.tolist()})")

    unet = UNetPitcher(**unet_cfg)
    model = ConsistencyPitcher(unet, sigma_data=0.5).to(device)

    state_dict = torch.load(config['models']['consistency_checkpoint'], map_location=device, weights_only=True)
    for key in list(state_dict.keys()):
        state_dict[key.replace('_orig_mod.', '')] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()

    if compile_model:
        model = torch.compile(model, backend=compile_backend)

    print(f"Model on device: {next(model.parameters()).device} (precision={precision} via autocast, compile={compile_model}, backend={compile_backend})")

    # Load vocoder
    hifigan, vocoder_type = load_vocoder(config, mel_cfg, device)

    print("Models loaded successfully")

    # Warmup pass to avoid cold start
    print("Warming up models...")
    import time
    dummy_mel = torch.randn(1, mel_cfg['n_mels'], 8, device=device)
    dummy_f0 = torch.randn(1, 8, device=device)

    use_autocast = dtype != torch.float32
    autocast_device = 'cuda' if use_gpu else 'cpu'
    warmup_start = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=autocast_device, dtype=dtype, enabled=use_autocast):
        current_input = torch.randn_like(dummy_mel)
        for step_num in range(len(chain_timesteps)):
            t_current = chain_timesteps[step_num].view(1)
            warm_out = model(x=current_input, t=t_current, mean=dummy_mel, f0=dummy_f0, noise_scheduler=noise_scheduler)
            if step_num < len(chain_timesteps) - 1:
                t_next = chain_timesteps[step_num + 1].view(1)
                fresh_noise = torch.randn_like(warm_out)
                current_input = noise_scheduler.add_noise(warm_out, fresh_noise, t_next)
        _ = hifigan(dummy_mel)
    if use_gpu:
        torch.cuda.synchronize()
    warmup_time = (time.perf_counter() - warmup_start) * 1000
    print(f"Warmup complete: {warmup_time:.1f}ms")

    # Streamer
    audio_processor = create_audio_processor(
        model, hifigan, noise_scheduler, device, config,
        chain_timesteps=chain_timesteps,
    )
    streamer = AudioStreamer(audio_callback=audio_processor)
    streamer.run()
