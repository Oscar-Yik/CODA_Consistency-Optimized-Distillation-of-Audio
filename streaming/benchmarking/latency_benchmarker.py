import numpy as np
import matplotlib.pyplot as plt
import time


class LatencyBenchmarker:
    def __init__(self, chunk_size, sample_rate, warmup_steps=20):
        self.hw_latencies = []
        self.inference_latencies = []
        self.total_latencies = []

        self.start_time = time.time()

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.warmup_steps = warmup_steps

    def add_latencies(self, hw_latency, inference_latency):
        self.hw_latencies.append(hw_latency)
        self.inference_latencies.append(inference_latency)
        self.total_latencies.append(hw_latency + inference_latency)

    def show_graph(self, save_path="streaming/benchmarking/latency_benchmarks.png"):
        buffer_budget = (self.chunk_size / self.sample_rate) * 1000  # max latency before audio breaks

        if len(self.total_latencies) <= self.warmup_steps:
            print(f"Not enough data to plot (only {len(self.total_latencies)} samples)")
            return

        # skip warmup iterations (python loads some stuff and takes a long time, it skews our stats)
        plot_total = self.total_latencies[self.warmup_steps:]
        plot_hw = self.hw_latencies[self.warmup_steps:]
        plot_inf = self.inference_latencies[self.warmup_steps:]

        # averages
        total_avg = np.mean(plot_total)
        inf_avg = np.mean(plot_inf)
        hw_avg = np.mean(plot_hw)

        plt.figure(figsize=(10, 5))

        # plot raw data
        plt.plot(plot_total, label="Total (HW + Inference)", color='#3498db')
        plt.plot(plot_hw, label=f'HW Baseline ({hw_avg:.2f}ms)', color='#2ecc71', alpha=0.6)
        plt.plot(plot_inf, label="Inference", color='#f1c40f', linewidth=2)

        # thresholds and averages
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
        plt.savefig(save_path)
        print(f"Latency graph saved to {save_path}")
