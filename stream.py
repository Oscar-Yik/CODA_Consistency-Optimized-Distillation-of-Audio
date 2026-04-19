import librosa
import pyaudio
import numpy as np
import collections
import time
import torch
import soundfile as sf
import yaml

from pitch_controller.models.unet import UNetPitcher
from pitch_controller.modules.BigVGAN.inference import load_model
from pitch_controller.utils import minmax_norm_diff, reverse_minmax_norm_diff
from streaming.fast_buffer import FastBuffer
from streaming.benchmarking.latency_benchmarker import LatencyBenchmarker
from utils import get_matched_f0, get_world_mel, log_f0

CHUNK = 512      
WINDOW_SIZE = 2048   
RAW_SAMPLE_RATE = 48000 # TODO: this is for my computers, may have to adjust for different machines  
TARGET_RATE = 24000
RAW_TO_TARGET = RAW_SAMPLE_RATE // TARGET_RATE

if RAW_SAMPLE_RATE % TARGET_RATE != 0:
    raise ValueError("RAW_SAMPLE_RATE must be a multiple of TARGET_RATE")

buffer = FastBuffer(WINDOW_SIZE)
benchmarker = LatencyBenchmarker(chunk_size=CHUNK, sample_rate=RAW_SAMPLE_RATE)

def preprocess_audio(audio_window):
    source_mel = get_world_mel(wav_path=None, sr=sr, wav=audio_window)
    f0_ref = get_matched_f0(x_wav=audio_window, y_wav=audio_window, method='world')
    f0_ref = log_f0(f0_ref, {'f0_bin': 345,
                             'f0_min': librosa.note_to_hz('C2'),
                             'f0_max': librosa.note_to_hz('C#6')})

    source_mel = torch.from_numpy(source_mel).float().unsqueeze(0).to(device)
    f0_ref = torch.from_numpy(f0_ref).float().unsqueeze(0).to(device)

    return source_mel, f0_ref


@torch.no_grad()
def audio_callback(in_data, frame_count, time_info, status):
    # time it takes before we even begin processing the audio data
    hw_latency = (time_info['output_buffer_dac_time'] - time_info['input_buffer_adc_time']) * 1000
    inference_start = time.perf_counter() # time to run the processing

    raw_audio_data = np.frombuffer(in_data, dtype=np.float32)
    downsampled_data = raw_audio_data[::RAW_TO_TARGET]
    context_window = buffer.add(downsampled_data)

    # preprocessing
    source_mel, f0_ref = preprocess_audio(context_window)
    source_x = minmax_norm_diff(source_mel, vmax=max_mel, vmin=min_mel)

    # inference
    output_wav = in_data # TODO: Placeholder for now
    # t_idx = torch.tensor([0], device=device)
    # model_output = model(x=source_x, mean=source_x, f0=f0_ref, t=t_idx, ref=None, embed=None)

    # pred_mel = reverse_minmax_norm_diff(model_output, vmax=max_mel, vmin=min_mel)
    # output_wav = hifigan(pred_mel)
    # output_wav = output_wav.squeeze().cpu().numpy().astype(np.float32)

    # Record latencies
    inference_time = (time.perf_counter() - inference_start) * 1000
    benchmarker.add_latencies(hw_latency, inference_time)

    return (output_wav, pyaudio.paContinue)


def run_realtime_inference():
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=RAW_SAMPLE_RATE,
                        input=True,
                        output=True,
                        frames_per_buffer=CHUNK,
                        stream_callback=audio_callback)

        print(f"* Stream started")
        stream.start_stream()

        while stream.is_active():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n* Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        benchmarker.show_graph()

if __name__ == '__main__':
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
    model = UNetPitcher(**unet_cfg)
    unet_path = 'ckpts/world_fixed_40.pt'

    state_dict = torch.load(unet_path, map_location=device, weights_only=True) # suppress security warning
    for key in list(state_dict.keys()):
        state_dict[key.replace('_orig_mod.', '')] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    if use_gpu:
        model.cuda()
    model.eval()

    #  load vocoder
    hifi_path = 'ckpts/bigvgan_24khz_100band/g_05000000.pt'
    hifigan, cfg = load_model(hifi_path, device=device)
    hifigan.eval()

    run_realtime_inference()

    