import torch
from scipy.ndimage import gaussian_filter1d
from utils import get_f0, get_autotuned_f0
import matplotlib.pyplot as plt
import numpy as np

def check_cuda():
    cuda_available = torch.cuda.is_available()

    print(f"Is CUDA available: {cuda_available}")
    if cuda_available:
        devices = [d for d in range(torch.cuda.device_count())]
        device_names = [torch.cuda.get_device_name(d) for d in devices]

        print("Available devices:")
        for device_num, device_name in zip(devices, device_names):
            print(f"{device_num}: {device_name}")

def compare_pitch_hybrid(wav_path, med_kernel=5, gauss_sigma=1.0):
    # 1. Extract Original F0
    f0_original = get_f0(wav_path, method='world', padding=False)
    
    # 2. Get your Autotuned F0 (already has a small medfilt internally)
    f0_autotuned = get_autotuned_f0(wav_path, wav=None, method='world', padding=False)
    
    # 3. Apply the Hybrid Smoothing (Median + Gaussian)
    # We take the autotuned stuff and run the Gaussian over it
    # sigma=1.0 is a "small" Gaussian; increase it for more blur
    f0_hybrid = gaussian_filter1d(f0_autotuned, sigma=gauss_sigma)
    
    # Setup Time Axis
    hop_length = 256
    sr = 24000
    time_axis = np.arange(len(f0_original)) * (hop_length / sr)
    
    # 4. Plotting the 3 Lines
    plt.figure(figsize=(15, 7))
    
    # Line 1: The Original Pitch (Raw)
    plt.plot(time_axis, f0_original, label='Original (Raw)', color='gray', alpha=0.3)
    
    # Line 2: The Snapped Autotune (Discrete steps)
    plt.plot(time_axis, f0_autotuned, label='Autotuned (Snapped)', color='blue', alpha=0.6)
    
    # Line 3: The Hybrid Version (Median + Gaussian)
    plt.plot(time_axis, f0_hybrid, label=f'Hybrid (Med + Gauss σ={gauss_sigma})', 
             color='darkorange', linewidth=2.5)
    
    # Focus the view on voiced regions
    voiced = f0_original > 0
    if np.any(voiced):
        plt.ylim(f0_original[voiced].min() - 30, f0_original[voiced].max() + 30)
    
    plt.title("Pitch Comparison: Raw vs Snapped vs Hybrid Smoothing")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_f0_comparison(wav_path1, wav_path2, label1="File 1", label2="File 2", sr=24000, hop_length=256):
    """
    Extracts and plots the F0 contours of two wav files for direct comparison.
    """
    # 1. Extract F0 for both files
    # Using 'world' method as seen in your utils.py for consistency
    f0_1 = get_f0(wav_path1, wav=None, method='world', padding=False)
    f0_2 = get_f0(wav_path2, wav=None, method='world', padding=False)
    
    # 2. Create time axes in seconds
    time1 = np.arange(len(f0_1)) * (hop_length / sr)
    time2 = np.arange(len(f0_2)) * (hop_length / sr)
    
    # 3. Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot first file
    plt.plot(time1, f0_1, label=label1, alpha=0.7, color='blue')
    
    # Plot second file
    plt.plot(time2, f0_2, label=label2, alpha=0.7, color='red', linestyle='--')
    
    # 4. Formatting the chart
    plt.title(f"F0 Pitch Comparison: {label1} vs {label2}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    
    # Focus the Y-axis on the audible range found in the files (ignore 0Hz/unvoiced)
    combined_f0 = np.concatenate([f0_1, f0_2])
    voiced_points = combined_f0[combined_f0 > 0]
    
    if len(voiced_points) > 0:
        plt.ylim(voiced_points.min() - 10, voiced_points.max() + 10)
    
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    # check_cuda()
    # compare_pitch_hybrid('examples/off-key.wav')
    plot_f0_comparison('examples/emma_twinkle.wav', 'output_template.wav')

if __name__ == "__main__":
    main()
