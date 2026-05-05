# Audio Samples

This folder contains audio samples of CODA output.

- `autotuned/`: This folder contains comparison between Diffpitcher and CODA. For both, we take the `off_tune` .wav file, and extract both the spectral envelope as well as the f0 pitch. We then apply a median and gaussian filter over the f0, and pitch snap it to the correct key. Then both models are given the spectral envelope as well as the smoothed and pitch-snapped f0 as input. The Diffpitcher model is run for 100 iterations, and we include the 1st, 2nd, 3rd, and 4th iterations of CODA.
- `no_autotune/`: This folder contains the same thing as the `autotuned/` folder, except the f0 is not modified at all. In other words, both models will try to generate the original off-tune audio.
- `streaming_coda_3_step_autotuned`: This folder contains samples from the streaming engine. Here, we first chunk the audio into different window sizes, and process (autotune + CODA inferrence) one chunk at a time. There are a lot of boundary artifacts in the generated audio compared to the previous. The streaming engine still needs more work before it can create good quality audio!