import json

import pyaudio
import numpy as np
import time
import soundfile as sf
import librosa

from streaming.fast_buffer import FastBuffer
from streaming.benchmarking.latency_benchmarker import LatencyBenchmarker


def _unpack_callback(result):
    """Allow the audio callback to return either `wav` or `(wav, components_dict)`."""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0], result[1]
    return result, None


class AudioStreamer:
    def __init__(self, audio_callback):
        self.audio_callback = audio_callback

        config_path='streaming/config.json'
        with open(config_path, 'r') as f:                                                                                                                                                                                                                              
            config = json.load(f) 
        self.config = config 

        audio_cfg = config['audio'] 
        self.chunk_size = audio_cfg['chunk_size']                                                                                                                                                                                                                      
        self.window_size = audio_cfg['window_size']                                                                                                                                                                                                                    
        self.raw_sample_rate = audio_cfg['raw_sample_rate']                                                                                                                                                                                                            
        self.target_sample_rate = audio_cfg['target_sample_rate']                                                                                                                                                                                                      
        self.downsample_ratio = self.raw_sample_rate // self.target_sample_rate

        source_cfg = config['source']                                                                                                                                                                                                                                  
        self.source_type = source_cfg['type']                                                                                                                                                                                                                          
        self.file_path = source_cfg.get('file_path') 

        if self.raw_sample_rate % self.target_sample_rate != 0:
            raise ValueError("raw_sample_rate must be a multiple of target_sample_rate")
        
        if self.source_type not in ['mic', 'file']:                                                                                                                                                                                                                    
            raise ValueError("source_type must be 'mic' or 'file'")      

        if self.source_type == 'file' and self.file_path is None:                                                                                                                                                                                                      
            raise ValueError("file_path is required when source_type='file'")    

        self.buffer = FastBuffer(self.window_size)
        self.benchmarker = LatencyBenchmarker(chunk_size=self.chunk_size, sample_rate=self.raw_sample_rate)

        # file streaming state
        self.file_audio_data = None
        self.file_position = 0
        self.file_original_sr = None

    def _load_audio_file(self):
        """Load and prepare audio file for streaming."""
        audio_data, sr = sf.read(self.file_path, dtype='float32')

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sr != self.raw_sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.raw_sample_rate)

        self.file_audio_data = audio_data
        self.file_original_sr = sr
        self.file_position = 0
        print(f"Loaded audio file: {self.file_path}")
        print(f"Duration: {len(audio_data) / self.raw_sample_rate:.2f}s, Sample rate: {sr} -> {self.raw_sample_rate}")

    def _mic_callback(self, in_data, frame_count, time_info, status):
        hw_latency = (time_info['output_buffer_dac_time'] - time_info['input_buffer_adc_time']) * 1000
        inference_start = time.perf_counter()

        raw_audio_data = np.frombuffer(in_data, dtype=np.float32)
        downsampled_data = raw_audio_data[::self.downsample_ratio]
        context_window = self.buffer.add(downsampled_data)

        output_wav, components = _unpack_callback(self.audio_callback(context_window))

        inference_time = (time.perf_counter() - inference_start) * 1000
        self.benchmarker.add_latencies(hw_latency, inference_time, components)

        if output_wav is not None:
            output_wav = librosa.resample(output_wav, orig_sr=self.target_sample_rate, target_sr=self.raw_sample_rate)
            if len(output_wav) != self.chunk_size:
                output_wav = output_wav[:self.chunk_size] if len(output_wav) > self.chunk_size else np.pad(output_wav, (0, self.chunk_size - len(output_wav)))
        else:
            output_wav = raw_audio_data

        return (output_wav.astype(np.float32), pyaudio.paContinue)

    def _stream_from_mic(self):
        """Stream audio from microphone."""
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paFloat32,
                          channels=1,
                          rate=self.raw_sample_rate,
                          input=True,
                          output=True,
                          frames_per_buffer=self.chunk_size,
                          stream_callback=self._mic_callback)

            print(f"* Streaming from microphone (rate={self.raw_sample_rate}Hz, chunk={self.chunk_size})")
            stream.start_stream()

            while stream.is_active():
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n* Stopping...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            self.benchmarker.show_graph()

    def _stream_from_file(self):
        """Stream audio from file, simulating real-time playback."""
        self._load_audio_file()

        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paFloat32,
                          channels=1,
                          rate=self.raw_sample_rate,
                          output=True,
                          frames_per_buffer=self.chunk_size)

            print(f"* Streaming from file (rate={self.raw_sample_rate}Hz, chunk={self.chunk_size})")

            chunk_duration = self.chunk_size / self.raw_sample_rate
            streamed_chunks = []

            while self.file_position < len(self.file_audio_data):
                chunk_start = time.perf_counter()

                # Extract chunk
                chunk_end = min(self.file_position + self.chunk_size, len(self.file_audio_data))
                raw_chunk = self.file_audio_data[self.file_position:chunk_end]

                # Pad if necessary
                if len(raw_chunk) < self.chunk_size:
                    raw_chunk = np.pad(raw_chunk, (0, self.chunk_size - len(raw_chunk)))

                # Downsample and add to buffer
                downsampled_chunk = raw_chunk[::self.downsample_ratio]
                context_window = self.buffer.add(downsampled_chunk)

                # inference
                inference_start = time.perf_counter()
                output_wav, components = _unpack_callback(self.audio_callback(context_window))
                inference_time = (time.perf_counter() - inference_start) * 1000

                # Upsample model output back to playback rate
                if output_wav is not None:
                    output_wav = librosa.resample(output_wav, orig_sr=self.target_sample_rate, target_sr=self.raw_sample_rate)
                    if len(output_wav) != self.chunk_size:
                        output_wav = output_wav[:self.chunk_size] if len(output_wav) > self.chunk_size else np.pad(output_wav, (0, self.chunk_size - len(output_wav)))
                else:
                    output_wav = raw_chunk

                # Play output
                output_wav = output_wav.astype(np.float32)
                stream.write(output_wav.tobytes())
                streamed_chunks.append(output_wav.copy())

                # Update position
                self.file_position = chunk_end

                # Record latency (no hardware latency for file streaming)
                self.benchmarker.add_latencies(0, inference_time, components)

                # Maintain real-time timing
                elapsed = time.perf_counter() - chunk_start
                sleep_time = max(0, chunk_duration - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n* Stopping...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

            if streamed_chunks:
                output_audio = np.concatenate(streamed_chunks)
                output_path = f"streaming/streamed_output_{int(time.time())}.wav"
                sf.write(output_path, output_audio, self.raw_sample_rate)
                print(f"* Wrote streamed audio to {output_path}")

            self.benchmarker.show_graph()

    def run(self):
        if self.source_type == 'mic':
            self._stream_from_mic()
        else:
            self._stream_from_file()
