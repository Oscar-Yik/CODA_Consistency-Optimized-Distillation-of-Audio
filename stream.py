import librosa
import pyaudio
import numpy as np
import collections
import time
import matplotlib.pyplot as plt
import torch
import soundfile as sf
import yaml

from pitch_controller.models.unet import UNetPitcher
from pitch_controller.modules.BigVGAN.inference import load_model
from pitch_controller.utils import minmax_norm_diff, reverse_minmax_norm_diff
from streaming.fast_buffer import FastBuffer
from utils import get_matched_f0, get_world_mel, log_f0

CHUNK = 512      
WINDOW_SIZE = 2048   
RAW_SAMPLE_RATE = 48000 # TODO: this is for mac, may have to adjust for different machines  
TARGET_RATE = 24000
RAW_TO_TARGET = RAW_SAMPLE_RATE // TARGET_RATE

if RAW_SAMPLE_RATE % RAW_TO_TARGET != 0:
    raise ValueError("RAW_SAMPLE_RATE must be a multiple of TARGET_RATE")

# Data collection for graphing
hw_latencies = []
inference_latencies = []
total_latencies = []
start_time = time.time()

buffer = FastBuffer(WINDOW_SIZE)

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
    global buffer

    # time it takes before we even begin processing the audio data
    hw_latency = (time_info['output_buffer_dac_time'] - time_info['input_buffer_adc_time']) * 1000
    hw_latencies.append(hw_latency)

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
    
    # more benchmarking
    inference_time = (time.perf_counter() - inference_start) * 1000
    total_latency = hw_latency + inference_time

    inference_latencies.append(inference_time)
    total_latencies.append(total_latency)
    
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
        show_graph()


def show_graph():
    warmup_steps = 20 # initial iterations that are slow because python initializes stuff. It skews our stats so we don't include it
    buffer_budget = (CHUNK / RAW_SAMPLE_RATE) * 1000 # if inference takes longer than this, the audio will "break"

    if len(total_latencies) > warmup_steps:
        plot_total = total_latencies[warmup_steps:]
        plot_hw = hw_latencies[warmup_steps:]
        plot_inf = inference_latencies[warmup_steps:]

        # averages
        total_avg = np.mean(plot_total)
        inf_avg = np.mean(plot_inf)
        hw_avg = np.mean(plot_hw)

        plt.figure(figsize=(10, 5))

        # raw data
        plt.plot(plot_total, label="Total (HW + Inference)", color='#3498db')
        plt.plot(plot_hw, label=f'HW Baseline ({hw_avg:.2f}ms)', color='#2ecc71', alpha=0.6)
        plt.plot(plot_inf, label="Inference", color='#f1c40f', linewidth=2)

        # inference time threshold
        plt.axhline(y=buffer_budget, color="#cc1f0c", linestyle=':', 
                    label=f'Buffer Budget ({buffer_budget:.2f}ms)')
        # averages
        plt.axhline(y=total_avg, color="#ee00ff", linestyle='--', 
                    label=f'Total Avg: {total_avg:.2f}ms')
        plt.axhline(y=inf_avg, color="#eb7c22", linestyle='--', 
                    label=f'Inference Avg: {inf_avg:.2f}ms')
        
        plt.title(f"Real-Time Pitch Correction Latencies (Skipping First {warmup_steps} Iterations)")
        plt.ylabel("Milliseconds")
        plt.xlabel("Buffer Iterations")
        plt.legend()
        plt.savefig("streaming/benchmarking/latency_benchmarks.png")
        plt.show()

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

    