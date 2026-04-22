import torch
from diffusers import DDIMScheduler

def verify_noise_order():
    # Initialize scheduler exactly like your training script
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    noise_scheduler.set_timesteps(50)

    print(f"{'Index':<8} | {'Timestep (t)':<15} | {'Clean Signal Retained (Alpha)':<30}")
    print("-" * 60)

    for i, t in enumerate(noise_scheduler.timesteps):
        # Alpha cumprod represents the mathematical percentage of the original x_0 
        # that survives the noise addition at timestep t.
        # 1.0 = Clean Audio. 0.0 = Pure Noise.
        alpha = noise_scheduler.alphas_cumprod[t].item()
        
        # Only print the edges to keep the output clean
        if i < 5 or i > 45:
            print(f"[{i:<4}]  | t = {t:<10} | {alpha*100:.2f}% clean signal")
        elif i == 5:
            print(f"{'':<8} | {'...':<15} | {'...':<30}")

if __name__ == "__main__":
    verify_noise_order()
