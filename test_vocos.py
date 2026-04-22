"""Sanity-check the Vocos vocoder in isolation: wav -> mel -> Vocos -> wav."""
import argparse

import librosa
import numpy as np
import soundfile as sf
import torch

from streaming.vocoder import VocosVocoder, resolve_dtype
from utils import get_mel, get_world_mel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='examples/emma_twinkle.wav')
    parser.add_argument('--output', default='examples/vocoder_test_vocos.wav')
    parser.add_argument('--mel', choices=['stft', 'world'], default='stft',
                        help="'stft' = plain librosa mel; 'world' = pyworld-resynth mel (matches stream.py)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()

    sr = 24000
    wav, _ = librosa.load(args.input, sr=sr)

    if args.mel == 'world':
        mel = get_world_mel(None, sr=sr, wav=wav)
    else:
        sf.write('/tmp/_mel_input.wav', wav, sr)
        mel = get_mel('/tmp/_mel_input.wav')
    print(f"mel shape={mel.shape}, range=[{mel.min():.2f}, {mel.max():.2f}]")

    dtype = resolve_dtype(args.precision)
    mel_t = torch.from_numpy(mel).float().unsqueeze(0).to(args.device)

    vocoder = VocosVocoder(device=args.device, compile_model=args.compile)
    autocast_device = 'cuda' if args.device == 'cuda' else 'cpu'
    with torch.no_grad(), torch.autocast(device_type=autocast_device, dtype=dtype, enabled=(dtype != torch.float32)):
        out = vocoder(mel_t)
    out = out.squeeze().cpu().numpy().astype(np.float32)

    sf.write(args.output, out, sr)
    print(f"wrote {args.output}  (peak={np.abs(out).max():.3f}, rms={np.sqrt((out**2).mean()):.4f})")


if __name__ == '__main__':
    main()
