import yaml
import numpy as np
import librosa
import soundfile as sf

import torch
from diffusers import DDIMScheduler

from pitch_controller.models.unet import UNetPitcher
from pitch_controller.models.consistency import ConsistencyPitcher
from pitch_controller.modules.BigVGAN.inference import load_model
from pitch_controller.utils import minmax_norm_diff, reverse_minmax_norm_diff
from utils import get_world_mel, get_matched_f0, log_f0

@torch.no_grad()
def coda(source, model, hifigan, device, steps=4, shift_semi=0):
    source_mel = get_world_mel(source, sr=sr)

    f0_ref = get_matched_f0(x=source, y=source, method='world', key="bb major")
    f0_ref = f0_ref * 2 ** (shift_semi / 12)

    f0_ref = log_f0(f0_ref, {'f0_bin': 345,
                             'f0_min': librosa.note_to_hz('C2'),
                             'f0_max': librosa.note_to_hz('C#6')})
    
    source_mel = torch.from_numpy(source_mel).float().unsqueeze(0).to(device)
    f0 = torch.from_numpy(f0_ref).float().unsqueeze(0).to(device)

    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    generator = torch.Generator(device=device).manual_seed(2024)

    noise_scheduler.set_timesteps(100)
    content_norm = minmax_norm_diff(source_mel, vmax=max_mel, vmin=min_mel)

    milestones = np.linspace(0, 100, steps, endpoint=False, dtype=int).tolist()


    # current_input = torch.randn_like(content_norm, generator=generator).to(device)
    current_input = torch.randn(content_norm.shape, generator=generator, device=device)
    
    for step_num, current_t_idx in enumerate(milestones):
        t_current = torch.as_tensor([noise_scheduler.timesteps[current_t_idx]], device=device)
        pred_x0 = model(current_input, t_current, content_norm, f0, noise_scheduler)
        
        if step_num < len(milestones) - 1:
            next_t_idx = milestones[step_num + 1]
            t_next = torch.as_tensor([noise_scheduler.timesteps[next_t_idx]], device=device)
            # fresh_noise = torch.randn_like(pred_x0, generator=generator)
            fresh_noise = torch.randn(pred_x0.shape, generator=generator, device=device)
            current_input = noise_scheduler.add_noise(pred_x0, fresh_noise, t_next)
        else:
            final_pred = pred_x0
            
    pred = reverse_minmax_norm_diff(final_pred, vmax=max_mel, vmin=min_mel)

    pred_audio = hifigan(pred)
    pred_audio = pred_audio.cpu().squeeze().clamp(-1, 1)

    return pred_audio

if __name__ == "__main__":
    min_mel = np.log(1e-5)
    max_mel = 2.5
    sr = 24000

    use_gpu = torch.cuda.is_available()
    device = 'cuda' if use_gpu else 'cpu'

    # load diffusion model
    config = yaml.load(open('pitch_controller/config/DiffWorld_24k.yaml'), Loader=yaml.FullLoader)
    mel_cfg = config['logmel']
    ddpm_cfg = config['ddpm']
    unet_cfg = config['unet']
    base_unet = UNetPitcher(**unet_cfg).to(device)
    model = ConsistencyPitcher(base_unet).to(device)
    coda_ckpt_path = "ckpts/coda_model_weights.pt"

    state = torch.load(coda_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    #  load vocoder
    hifi_path = 'ckpts/bigvgan_24khz_100band/g_05000000.pt'
    hifigan, cfg = load_model(hifi_path, device=device)
    hifigan.eval()

    pred_audio = coda('examples/emma_twinkle.wav', model, hifigan, device, steps=4, shift_semi=0)
    sf.write('output_emma_twinkle.wav', pred_audio, samplerate=sr)
