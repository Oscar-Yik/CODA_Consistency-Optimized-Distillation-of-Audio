import numpy as np
import matplotlib.pyplot as plt
import time


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
                   save_path="streaming/benchmarking/latency_benchmarks.png",
                   inference_save_path="streaming/benchmarking/inference_benchmarks.png"):
        buffer_budget = (self.chunk_size / self.sample_rate) * 1000  # max latency before audio breaks

        if len(self.total_latencies) <= self.warmup_steps:
            print(f"Not enough data to plot (only {len(self.total_latencies)} samples)")
            return

        # ---------- Graph 1: total vs hw vs inference ----------
        plot_total = self._post_warmup(self.total_latencies)
        plot_hw = self._post_warmup(self.hw_latencies)
        plot_inf = self._post_warmup(self.inference_latencies)

        total_avg = np.mean(plot_total)
        inf_avg = np.mean(plot_inf)
        hw_avg = np.mean(plot_hw)

        plt.figure(figsize=(10, 5))
        plt.plot(plot_total, label="Total (HW + Inference)", color='#3498db')
        plt.plot(plot_hw, label=f'HW Baseline ({hw_avg:.2f}ms)', color='#2ecc71', alpha=0.6)
        plt.plot(plot_inf, label="Inference", color='#f1c40f', linewidth=2)

        plt.axhline(y=buffer_budget, color="#cc1f0c", linestyle=':',
                    label=f'Buffer Budget ({buffer_budget:.2f}ms)')
        plt.axhline(y=total_avg, color="#ee00ff", linestyle='--',
                    label=f'Total Avg: {total_avg:.2f}ms')
        plt.axhline(y=inf_avg, color="#eb7c22", linestyle='--',
                    label=f'Inference Avg: {inf_avg:.2f}ms')

        plt.title(f"Real-Time Pitch Correction Latencies (Skipping First {self.warmup_steps} Iterations)")
        plt.ylabel("Milliseconds")
        plt.xlabel("Buffer Iterations")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Latency graph saved to {save_path}")

        # ---------- Graph 2: inference breakdown ----------
        plot_pre = self._post_warmup(self.preprocess_latencies)
        plot_model = self._post_warmup(self.model_latencies)
        plot_voc = self._post_warmup(self.vocoder_latencies)

        if not any(plot_pre) and not any(plot_model) and not any(plot_voc):
            print("Skipping inference breakdown graph (no component timings recorded).")
            return

        pre_avg = np.mean(plot_pre)
        model_avg = np.mean(plot_model)
        voc_avg = np.mean(plot_voc)

        plt.figure(figsize=(10, 5))
        plt.plot(plot_inf, label="Total Inference", color='#f1c40f', linewidth=2)
        plt.plot(plot_pre, label="Preprocess", color='#1abc9c', alpha=0.8)
        plt.plot(plot_model, label="Model", color='#9b59b6', alpha=0.8)
        plt.plot(plot_voc, label="Vocoder", color='#34495e', alpha=0.8)

        plt.axhline(y=buffer_budget, color="#cc1f0c", linestyle=':',
                    label=f'Buffer Budget ({buffer_budget:.2f}ms)')
        plt.axhline(y=inf_avg, color='#f1c40f', linestyle='--',
                    label=f'Inference Avg: {inf_avg:.2f}ms')
        plt.axhline(y=pre_avg, color='#1abc9c', linestyle='--',
                    label=f'Preprocess Avg: {pre_avg:.2f}ms')
        plt.axhline(y=model_avg, color='#9b59b6', linestyle='--',
                    label=f'Model Avg: {model_avg:.2f}ms')
        plt.axhline(y=voc_avg, color='#34495e', linestyle='--',
                    label=f'Vocoder Avg: {voc_avg:.2f}ms')

        plt.title(f"Inference Breakdown (Skipping First {self.warmup_steps} Iterations)")
        plt.ylabel("Milliseconds")
        plt.xlabel("Buffer Iterations")
        plt.legend()
        plt.tight_layout()
        plt.savefig(inference_save_path)
        plt.close()
        print(f"Inference breakdown graph saved to {inference_save_path}")
