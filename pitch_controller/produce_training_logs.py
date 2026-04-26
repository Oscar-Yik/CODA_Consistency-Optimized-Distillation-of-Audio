import csv
import matplotlib.pyplot as plt

def plot_eval_metrics(csv_path, save_path):
    epochs = []
    mel_mses = []
    f0_maes = []
    blur_ratios = []
    mel_mses_full = []
    
    print(f"Reading evaluation data from {csv_path}...")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            mel_mses.append(float(row['mel_mse']))
            blur_ratios.append(float(row['blur_ratio']))
            mel_mses_full.append(float(row['mel_mse_full']))
            
            # Handle cases where F0 tracking might have failed (saved as empty or None)
            f0_val = row['f0_mae']
            if f0_val and f0_val != 'None':
                f0_maes.append(float(f0_val))
            else:
                f0_maes.append(None) # Keep list aligned

    _, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]
    
    # --- Top Subplot: Mel Spectrogram MSE ---
    ax1.plot(epochs, mel_mses, marker='o', color='tab:blue', linewidth=2, label='Mel MSE t_idx = 50')
    ax1.set_title('Validation Mel-Spectrogram t_idx = 50 Error (Timbre/Texture)')
    ax1.set_ylabel('MSE')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # --- Middle Subplot: Blur Ratio ---
    ax2.plot(epochs, blur_ratios, marker='o', color='tab:red', linewidth=2, label='Blur Ratio')
    ax2.set_title('Blur ratio (Average Blur)')
    ax2.set_ylabel('Blur Ratio')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # --- Bottom Subplot: F0 Pitch Tracking MAE ---
    # Filter out None values just for plotting the line
    valid_epochs = [e for e, f in zip(epochs, f0_maes) if f is not None]
    valid_f0s = [f for f in f0_maes if f is not None]
    
    ax3.plot(valid_epochs, valid_f0s, marker='s', color='tab:orange', linewidth=2, label='F0 MAE (Hz)')
    ax3.set_title('F0 Pitch Tracking Error (Melody Accuracy)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Mean Absolute Error (Hz)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()

    ax4.plot(epochs, mel_mses_full, marker='o', color='tab:blue', linewidth=2, label='Mel MSE t_idx = 0')
    ax4.set_title('Validation Mel-Spectrogram t_idx = 0 Error (Timbre/Texture)')
    ax4.set_ylabel('MSE')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150) # High resolution for reports
    print(f"Evaluation graphs saved to {save_path}!")

if __name__ == "__main__":

    # Mel MSE and f0 MAE Graphs
    csv_file = 'logs_consistency/eval_metrics.csv' 
    output_image = 'logs_consistency/eval_metrics_graph.png'
    
    plot_eval_metrics(csv_file, output_image)
