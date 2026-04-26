import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import pandas as pd

class LatencyBenchmarker:
    def __init__(self, chunk_size, sample_rate, warmup_steps=20):
        self.hw_latencies = []
        self.inference_latencies = []
        self.total_latencies = []

        # Per-component inference breakdown (aligned with inference_latencies).
        self.preprocess_latencies = []
        self.model_latencies = []
        self.vocoder_latencies = []

        self.start_time = time.time()

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.warmup_steps = warmup_steps

    def add_latencies(self, hw_latency, inference_latency, components=None):
        self.hw_latencies.append(hw_latency)
        self.inference_latencies.append(inference_latency)
        self.total_latencies.append(hw_latency + inference_latency)

        components = components or {}
        self.preprocess_latencies.append(components.get('preprocess', 0.0))
        self.model_latencies.append(components.get('model', 0.0))
        self.vocoder_latencies.append(components.get('vocoder', 0.0))

    def _post_warmup(self, series):
        return series[self.warmup_steps:]

    def show_graph(self,
                   save_path="streaming/benchmarking/latency_benchmarks_seaborn.pdf",
                   inference_save_path="streaming/benchmarking/inference_benchmarks_seaborn.pdf"):
        buffer_budget = (self.chunk_size / self.sample_rate) * 1000  # max latency before audio breaks

        if len(self.total_latencies) <= self.warmup_steps:
            print(f"Not enough data to plot (only {len(self.total_latencies)} samples)")
            return

        # Apply Seaborn's modern theme
        sns.set_theme(style="darkgrid", context="talk")

        # ---------- Graph 1: Total vs HW vs Inference ----------
        plot_total = self._post_warmup(self.total_latencies)
        plot_hw = self._post_warmup(self.hw_latencies)
        plot_inf = self._post_warmup(self.inference_latencies)

        total_avg = np.mean(plot_total)
        inf_avg = np.mean(plot_inf)
        hw_avg = np.mean(plot_hw)

        # Package data into a Pandas DataFrame for Seaborn
        df1 = pd.DataFrame({
            'Total (HW + Inference)': plot_total,
            'Inference': plot_inf,
            'HW Baseline': plot_hw
        })

        plt.figure(figsize=(12, 6))
        
        # Use a high-contrast 'husl' palette for the main pipeline metrics
        palette1 = sns.color_palette("husl", 3)
        sns.lineplot(data=df1, palette=palette1, linewidth=2.5, alpha=0.9)

        # Add horizontal threshold/average lines
        plt.axhline(y=buffer_budget, color="red", linestyle=':', linewidth=2.5,
                    label=f'Buffer Budget ({buffer_budget:.2f}ms)')
        plt.axhline(y=total_avg, color=palette1[0], linestyle='--', alpha=0.7,
                    label=f'Total Avg: {total_avg:.2f}ms')
        plt.axhline(y=inf_avg, color=palette1[1], linestyle='--', alpha=0.7,
                    label=f'Inference Avg: {inf_avg:.2f}ms')

        plt.title(f"Real-Time Pitch Correction Latencies (Skipping First {self.warmup_steps} Iterations)", fontweight='bold', pad=15)
        plt.ylabel("Latency (Milliseconds)", fontweight='bold')
        plt.xlabel("Buffer Iterations", fontweight='bold')
        
        # Clean up legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
        sns.despine(left=True, bottom=True)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Latency graph saved to {save_path}")

        # ---------- Graph 2: Inference Breakdown ----------
        plot_pre = self._post_warmup(self.preprocess_latencies)
        plot_model = self._post_warmup(self.model_latencies)
        plot_voc = self._post_warmup(self.vocoder_latencies)

        if not any(plot_pre) and not any(plot_model) and not any(plot_voc):
            print("Skipping inference breakdown graph (no component timings recorded).")
            return

        pre_avg = np.mean(plot_pre)
        model_avg = np.mean(plot_model)
        voc_avg = np.mean(plot_voc)

        # Package data into a Pandas DataFrame
        df2 = pd.DataFrame({
            'Total Inference': plot_inf,
            'Model': plot_model,
            'Vocoder': plot_voc,
            'Preprocess': plot_pre
        })

        plt.figure(figsize=(12, 6))
        
        # Use 'Set2' for distinctly separating the sub-components
        palette2 = sns.color_palette("Set2", 4)
        sns.lineplot(data=df2, palette=palette2, linewidth=2.5, alpha=0.9)

        # Add horizontal threshold/average lines
        plt.axhline(y=buffer_budget, color="red", linestyle=':', linewidth=2.5,
                    label=f'Buffer Budget {buffer_budget:.2f}ms')
        plt.axhline(y=inf_avg, color=palette2[0], linestyle='--', alpha=0.7,
                    label=f'Inference Avg: {inf_avg:.2f}ms')
        plt.axhline(y=model_avg, color=palette2[1], linestyle='--', alpha=0.7,
                    label=f'Model Avg: {model_avg:.2f}ms')
        plt.axhline(y=voc_avg, color=palette2[2], linestyle='--', alpha=0.7,
                    label=f'Vocoder Avg: {voc_avg:.2f}ms')
        plt.axhline(y=pre_avg, color=palette2[3], linestyle='--', alpha=0.7,
                    label=f'Preprocess Avg: {pre_avg:.2f}ms')

        print(f"Inference Avg: {inf_avg}")

        plt.title(f"Streaming Inference Component Breakdown", fontweight='bold', pad=15)
        plt.ylabel("Latency (Milliseconds)", fontweight='bold')
        plt.xlabel("Buffer Iterations", fontweight='bold')
        
        # Move legend outside the plot so it doesn't cover data lines
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
        sns.despine(left=True, bottom=True)
        
        plt.tight_layout()
        plt.savefig(inference_save_path, bbox_inches='tight')
        plt.close()
        print(f"Inference breakdown graph saved to {inference_save_path}")
