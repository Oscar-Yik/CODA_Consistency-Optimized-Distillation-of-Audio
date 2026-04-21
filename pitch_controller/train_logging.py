import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

def plot_eval_metrics(csv_path, save_path):
    epochs = []
    mel_mses = []
    f0_maes = []
    blur_ratios = []
    
    print(f"Reading evaluation data from {csv_path}...")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            mel_mses.append(float(row['mel_mse']))
            blur_ratios.append(float(row['blur_ratio']))
            
            # Handle cases where F0 tracking might have failed (saved as empty or None)
            f0_val = row['f0_mae']
            if f0_val and f0_val != 'None':
                f0_maes.append(float(f0_val))
            else:
                f0_maes.append(None) # Keep list aligned

    # Create a figure with 3 subplots (stacked vertically)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # --- Top Subplot: Mel Spectrogram MSE ---
    ax1.plot(epochs, mel_mses, marker='o', color='tab:blue', linewidth=2, label='Mel MSE')
    ax1.set_title('Validation Mel-Spectrogram Error (Timbre/Texture)')
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

    plt.tight_layout()
    plt.savefig(save_path, dpi=150) # High resolution for reports
    print(f"Evaluation graphs saved to {save_path}!")

def plot_trajectory_loss(csv_path, save_path, sample_every=5):
    # Dictionary structure: data[epoch][t_idx] = [loss1, loss2, ...]
    epoch_data = defaultdict(lambda: defaultdict(list))
    
    print(f"Reading data from {csv_path}...")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row['epoch'])
            t_idx = int(row['t_idx'])
            loss = float(row['loss'])
            epoch_data[epoch][t_idx].append(loss)
            
    plt.figure(figsize=(12, 8))
    
    # Sort the epochs so we plot them in chronological order
    epochs = sorted(epoch_data.keys())
    
    # Create a colormap (e.g., viridis) to show progression from early (purple) to late (yellow)
    colors = cm.viridis(np.linspace(0, 1, len(epochs)))
    
    lines_plotted = 0
    for i, epoch in enumerate(epochs):
        # We only plot every Nth epoch (and always plot the very last one) to avoid clutter
        if epoch % sample_every != 0 and epoch != epochs[-1]:
            continue
            
        t_losses = epoch_data[epoch]
        t_indices = sorted(t_losses.keys())
        
        # Calculate the mean loss for this specific epoch at each timestep
        avg_losses = [np.mean(t_losses[t]) for t in t_indices]
        
        plt.plot(t_indices, avg_losses, marker='.', linestyle='-', 
                 color=colors[i], alpha=0.8, label=f'Epoch {epoch}')
        lines_plotted += 1

    plt.yscale('log')
    plt.title('Consistency Distillation: Loss Trajectory Over Time')
    plt.xlabel('Timestep Index (t_idx)')
    plt.ylabel('MSE Loss (Log Scale)')
    
    # Put the legend outside the graph so it doesn't cover the lines
    if lines_plotted <= 25:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}!")

if __name__ == "__main__":
    # Change sample_every=1 if you want to see literally every single epoch
    # file_name = "trajectory_loss_1776497415.0009303"

    # plot_trajectory_loss(
    #     f'logs_consistency/{file_name}.csv', 
    #     f'logs_consistency/{file_name}.png',
    #     sample_every=3
    # )

    # Mel MSE and f0 MAE Graphs
    csv_file = 'logs_consistency/eval_metrics_1776671413.5251737.csv' 
    output_image = 'logs_consistency/eval_metrics_graph.png'
    
    # Uncomment to run:
    plot_eval_metrics(csv_file, output_image)
