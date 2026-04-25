import numpy as np
import torch
import librosa
from librosa.core import load
import matplotlib.pyplot as plt
import pysptk
import pyworld as pw
from fastdtw import fastdtw
from scipy import spatial
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import os
from tqdm import tqdm
import soundfile as sf
import time

from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(sr=24000, n_fft=1024, n_mels=100, fmin=0, fmax=12000)


def _get_best_mcep_params(fs):
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        raise ValueError(f"Not found the setting for {fs}.")


def get_mel(wav_path, wav=None):
    if wav.all() == None:
        wav, _ = load(wav_path, sr=24000)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    if mel_spectrogram.shape[-1] % 8 != 0:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, 8 - mel_spectrogram.shape[-1] % 8)), 'minimum')

    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram


def get_world_mel(wav_path=None, sr=24000, wav=None):
    start_time = time.time()

    if wav_path is not None:
        wav, _ = librosa.load(wav_path, sr=24000)
    # wav = (wav * 32767).astype(np.int16)
    # wav = (wav / 32767).astype(np.float64)
    wav = wav.astype(np.float64)
    wav = wav[:(wav.shape[0] // 256) * 256]

    _f0, t = pw.dio(wav, sr, frame_period=256/sr*1000) # using a fixed frame period improves performance
    # _f0, t = pw.dio(wav, sr)
    f0 = pw.stonemask(wav, _f0, t, sr)
    sp = pw.cheaptrick(wav, f0, t, sr)
    ap = pw.d4c(wav, f0, t, sr)
    # wav_hat = pw.synthesize(f0 * 0, sp, ap, sr)
    wav_hat = pw.synthesize(f0 * 0, sp, ap, sr, frame_period=256/sr*1000)

    # pyworld output does not pad left
    wav_hat = wav_hat[:len(wav)]
    # wav_hat = wav_hat[256//2: len(wav)+256//2]
    assert len(wav_hat) == len(wav)
    wav = wav_hat.astype(np.float32)
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    if mel_spectrogram.shape[-1] % 8 != 0:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, 8 - mel_spectrogram.shape[-1] % 8)), 'minimum')

    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    # print(f"mel duration: {time.time() - start_time}")
    return log_mel_spectrogram


def get_f0(wav_path, wav, method='pyin', padding=True, f0_min_note=None, f0_max_note=None, fast_f0=False):
    sr = 24000

    if wav is None:
        if wav_path is not None:
            wav, sr = load(wav_path, sr=sr)
        else:
            raise ValueError("Must provide either 'wav' (array) or 'wav_path' (string)")
        
    f0_min_note = f0_min_note if f0_min_note != None else "C2"
    f0_max_note = f0_max_note if f0_max_note != None else "C#6"
        
    if method == 'pyin':
        wav = wav[:(wav.shape[0] // 256) * 256]
        wav = np.pad(wav, 384, mode='reflect')
        if fast_f0:
            f0 = librosa.yin(wav, frame_length=1024, hop_length=256, center=False, sr=24000,
                                    fmin=librosa.note_to_hz(f0_min_note),
                                    fmax=librosa.note_to_hz(f0_max_note))
            energy = np.array([np.sqrt(np.mean(wav[i*256:i*256+1024]**2)) for i in range(len(f0))])                                                                                                                                                                                           
            f0[energy < 1e-3] = 0 
        else: 
             f0, _, _ = librosa.pyin(wav, frame_length=1024, hop_length=256, center=False, sr=24000,
                                fmin=librosa.note_to_hz(f0_min_note),
                                fmax=librosa.note_to_hz(f0_max_note), fill_na=0)
            
    elif method == 'world':
        wav = (wav * 32767).astype(np.int16)
        wav = (wav / 32767).astype(np.float64)
        _f0, t = pw.dio(wav, fs=sr, frame_period=256/sr*1000,
                        f0_floor=librosa.note_to_hz(f0_min_note),
                        f0_ceil=librosa.note_to_hz(f0_max_note))
        f0 = pw.stonemask(wav, _f0, t, sr)
        f0 = f0[:-1]

    if padding is True:
        if f0.shape[-1] % 8 !=0:
            f0 = np.pad(f0, ((0, 8-f0.shape[-1] % 8)), 'constant', constant_values=0)

    return f0


def get_mcep(x_path, x_wav, n_fft=1024, n_shift=256, sr=24000):
    if x_wav is None:
        if x_path is not None:
            x, _ = librosa.load(x, sr=sr)
        else:
            raise ValueError("Must provide either 'x_path' (string) or 'x_wav' (array)")
    else:
        x = x_wav
        
    n_frame = (x.shape[0] // 256)
    x = np.pad(x, 384, mode='reflect')
    # n_frame = (len(x) - n_fft) // n_shift + 1
    win = pysptk.sptk.hamming(n_fft)
    mcep_dim, mcep_alpha = _get_best_mcep_params(sr)
    mcep = [pysptk.mcep(x[n_shift * i: n_shift * i + n_fft] * win,
                        mcep_dim, mcep_alpha,
                        eps=1e-6, etype=1,)
            for i in range(n_frame)
            ]
    mcep = np.stack(mcep)
    return mcep


def get_matched_f0(x=None, y=None, x_wav=None, y_wav=None, method='world', n_fft=1024, n_shift=256, key=None, f0_min_note=None, f0_max_note=None, fast_f0=False):
    # f0_x = get_f0(x, method='pyin', padding=False)
    # f0_y = get_f0(y, method=method, padding=False)
    f0_y = get_autotuned_f0(y, y_wav, method, False, key=key, f0_min_note=f0_min_note, f0_max_note=f0_max_note, fast_f0=fast_f0)
    # print(f0_y.max())
    # print(f0_y.min())

    # -------- Code commented below is dtw + other matching stuff. For our purposes, x and y are the same so no need to perform matching ------
    
    # mcep_x = get_mcep(x, x_wav, n_fft=n_fft, n_shift=n_shift)
    # mcep_y = get_mcep(y, y_wav, n_fft=n_fft, n_shift=n_shift)

    # _, path = fastdtw(mcep_x, mcep_y, dist=spatial.distance.euclidean)
    # twf = np.array(path).T
    # # f0_x = gen_mcep[twf[0]]
    # nearest = []
    # for i in range(len(f0_y)):
    #     idx = np.argmax(1 * twf[0] == i)
    #     nearest.append(twf[1][idx])

    # f0_y = f0_y[nearest]

    # f0_y = f0_y.astype(np.float32)

    if f0_y.shape[-1] % 8 != 0:
        f0_y = np.pad(f0_y, ((0, 8 - f0_y.shape[-1] % 8)), 'constant', constant_values=0)

    return f0_y


def f0_to_coarse(f0, hparams):

    f0_bin = hparams['f0_bin']
    f0_max = hparams['f0_max']
    f0_min = hparams['f0_min']
    is_torch = isinstance(f0, torch.Tensor)
    # to mel scale
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)

    unvoiced = (f0_mel == 0)

    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1

    f0_mel[unvoiced] = 0

    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 0, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse


def log_f0(f0, hparams):
    f0_bin = hparams['f0_bin']
    f0_max = hparams['f0_max']
    f0_min = hparams['f0_min']

    f0_mel = np.zeros_like(f0)
    f0_mel[f0 != 0] = 12*np.log2(f0[f0 != 0]/f0_min) + 1
    f0_mel_min = 12*np.log2(f0_min/f0_min) + 1
    f0_mel_max = 12*np.log2(f0_max/f0_min) + 1

    unvoiced = (f0_mel == 0)

    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1

    f0_mel[unvoiced] = 0

    f0_coarse = np.rint(f0_mel).astype(int)
    assert f0_coarse.max() <= (f0_bin-1) and f0_coarse.min() >= 0, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse


def show_plot(tensor):
    tensor = tensor.squeeze().cpu()
    # plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.show()


def _get_key_notes(key=None):
    """Returns a set of valid MIDI note classes (0-11) for the given key.
    If key is None, returns all 12 chromatic notes."""

    if key is None:
        return set(range(12))

    # Major and minor scale intervals (in semitones from root)
    scales = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'minor': [0, 2, 3, 5, 7, 8, 10]
    }

    # Parse key string (e.g., "C", "C#", "Db", "C major", "A minor")
    key = key.strip().lower()

    # Extract scale type
    if 'major' in key:
        scale = scales['major']
        key = key.replace('major', '').strip()
    elif 'minor' in key or 'min' in key:
        scale = scales['minor']
        key = key.replace('minor', '').replace('min', '').strip()
    else:
        scale = scales['major']  # Default to major

    # Parse root note (C, C#, Db, etc.)
    note_map = {
        'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3,
        'e': 4, 'f': 5, 'f#': 6, 'gb': 6, 'g': 7, 'g#': 8,
        'ab': 8, 'a': 9, 'a#': 10, 'bb': 10, 'b': 11
    }

    if key not in note_map:
        raise ValueError(f"Invalid key: {key}. Use format like 'C', 'C# major', 'A minor', etc.")

    root = note_map[key]
    return set((root + interval) % 12 for interval in scale)


def get_autotuned_f0(wav_path, wav, method, padding, key=None, f0_min_note=None, f0_max_note=None, fast_f0=False):
    f0_raw = get_f0(wav_path, wav, method=method, padding=padding, f0_min_note=f0_min_note, f0_max_note=f0_max_note, fast_f0=fast_f0)

    voiced_mask = f0_raw > 0
    autotuned_f0 = np.copy(f0_raw)

    if np.any(voiced_mask):
        midi_notes = librosa.hz_to_midi(f0_raw[voiced_mask])

        if key is None:
            # Original behavior: snap to nearest chromatic note
            snapped_midi = np.round(midi_notes)
        else:
            # Snap to nearest note in the specified key
            valid_notes = _get_key_notes(key)
            snapped_midi = np.zeros_like(midi_notes)

            for i, note in enumerate(midi_notes):
                octave = int(note // 12)
                # Find nearest valid note in the key
                note_class = note % 12
                valid_note_classes = np.array(list(valid_notes))
                distances = np.abs(valid_note_classes - note_class)

                # Handle wraparound (e.g., B to C)
                distances = np.minimum(distances, 12 - distances)
                nearest_idx = np.argmin(distances)
                nearest_note_class = valid_note_classes[nearest_idx]

                # Check if we need to go up or down an octave
                if note_class - nearest_note_class > 6:
                    octave += 1
                elif nearest_note_class - note_class > 6:
                    octave -= 1

                snapped_midi[i] = octave * 12 + nearest_note_class

        autotuned_f0[voiced_mask] = librosa.midi_to_hz(snapped_midi)

    # Adaptive median filter - use largest valid odd kernel size
    num_frames = len(autotuned_f0)
    desired_kernel = 51

    if num_frames >= desired_kernel:
        kernel_size = desired_kernel
    elif num_frames >= 3:  # minimum kernel size for median filter
        # Use largest odd number <= num_frames
        kernel_size = num_frames if num_frames % 2 == 1 else num_frames - 1
    else:
        kernel_size = None  # too small to filter

    if kernel_size is not None:
        autotuned_f0 = signal.medfilt(autotuned_f0, kernel_size=kernel_size)

    hybrid_f0 = gaussian_filter1d(autotuned_f0, sigma=2.0)

    return hybrid_f0


def apply_complex_degradation(y, sr, base_shift):
    # Apply the base constant offset (e.g., -1.2 semitones)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=base_shift)
    
    # Apply Jitter (Pitch Instability)
    # Simulate this by breaking the audio into frames and shifting each frame slightly differently.
    hop_length = 512
    frames = librosa.util.frame(y_shifted, frame_length=2048, hop_length=hop_length)
    
    # Create a slow-moving jitter curve (0.5Hz to 2Hz)
    t = np.linspace(0, len(y_shifted)/sr, frames.shape[1])
    jitter_curve = 0.3 * np.sin(2 * np.pi * 1.5 * t) # 1.5Hz oscillation, 0.3 semitone depth
    
    y_jittered = np.zeros_like(y_shifted)
    
    # Apply the varying shift frame-by-frame
    for i in range(frames.shape[1]):
        start = i * hop_length
        end = start + 2048
        if end > len(y_shifted): break
        
        # Shift the current frame by the jitter value
        frame_shifted = librosa.effects.pitch_shift(frames[:, i], sr=sr, n_steps=jitter_curve[i])
        y_jittered[start:end] += frame_shifted * np.hanning(2048) # Overlap-add with windowing
        
    return y_jittered


def generate_dataset(input_dir, output_dir, sr=24000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "off_tune"))
        os.makedirs(os.path.join(output_dir, "ground_truth"))

    files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    for f in tqdm(files):
        y, _ = librosa.load(os.path.join(input_dir, f), sr=sr)
        
        sf.write(os.path.join(output_dir, "ground_truth", f), y, sr)
    
        random_shift = np.random.uniform(-1.5, 1.5)
        y_off = apply_complex_degradation(y, sr, random_shift)
        
        sf.write(os.path.join(output_dir, "off_tune", f), y_off, sr)


if __name__ == '__main__':
    mel = get_mel('target.wav')
    f0 = get_f0('target.wav')