# Streaming
## How to run
`uv pip install -r <your requirements file>`

## Config
All configuration is inside of `streaming/config.json`

### Chunk and Window size
```json
"audio": {
    "chunk_size": 4096,
    "window_size": 2048,
    "raw_sample_rate": 48000,
    "target_sample_rate": 24000
},
```
My computer's native sample rate is 48k; adjust yours if it differs. We downsample this input to a 24k target rate. Note that `chunk_siz`e is defined relative to the `raw_sample_rate`, while `window_size` is relative to the `target_sample_rate`. Under this configuration, the window_size is numerically half of the chunk_size, but they represent the exact same temporal duration. The chunk size is how much data we buffer before actually processing it, and the window size is how much data we process each time we get a new `chunk_size` of data. 


In actual autotune, what you would do is have a larger `window_size` temporal duration compared to `chunk_size`, so that the model output would be "smoother". People would also fade in a new window (by overlapping the begining of that window with the previous window) to further make the audio less glitchy. However, unfortunatly our thing is too slow so we can't do that (lmao) but if we manage to have some extra buffer space then we can consider making the `window_size` bigger. Anyways, what I'm trying to say is just always keep the `window_size` as half of `chunk_size`. Lowering these will 1. make the delay between when you sing something and when the computer outputs something shorter and 2. make the quality worse. Feel free to tweak these settings to see what works best.

### Source
```json
"source": {
    "type": "file",
    "file_path": "examples/emma_twinkle.wav"
},
```
`type` is either "mic" or "file". if you choose file, then a `file_path` pointing to a .wav file must be specified. If you choose "mic", `file_path` will be ignored.

### Performance
```json
"performance": {
    "precision": "fp32",
    "compile": false,
    "compile_backend": "inductor",
    "consistency_iterations": 3,
    "chain_indices": [0, 30, 60],
    "fast_f0": true
}
```
- `precision`: one of `["fp32", "fp16", "bf16"]`
- `compile`: `true` or `false`. Determinines if we run `torch.compile` on the model and vocoder
- `compile_backend`: the backend used for compilation. I'm pretty sure "inductor" is correct for like your GPU, but if something breaks then it might be because of this 
- `consistency_iterations`: the number of iterations we run the consistency model
- `chain_indices`: same as the chain indices in test_consistency.py. The length must match `consistency_iterations`. Leave this blank for the script to use evenly-spaced indices.
- `fast_f0`: This decides if we use `pyin` or `yin` to get the f0. `pyin` uses HMM (probabilistic) so it is slow, but performs better. If you want to use it, it would be good to configure the min and max f0s as explained in the next section. `yin` is a lot faster as it is deterministic, but the quality is slightly worse. Overall, our audio quality is already bad due to the choppiness of the audio from processing tiny chunks at a time. So I don't think using the better one makes thaaaat big of a difference, so here the default is to use `yin` instead.

### Pitch
```json
"pitch": {
    "f0_bin": 345,
    "f0_min_note": "C2",
    "f0_max_note": "C#6",
    "key": "bb major"
},
```
The key can be configured to pitch snap only to notes within the key. Leave it as an empty string to pitch snap to the closest note instead. The string follows the pattern of `"<note><#/b/nothing> <major/minor>"`. min and max f0 range can be narrowed to make the preprocessing faster. But since it is already lighting fast, it doesn't matter that much. Leaving it as it is will be fine for performance.

## Running only the Vocoder
Since we are using a smaller vocoder, the output may contain artifacts. If running `stream.py` pproduces bad quality audio and you want to see if it is the vocoder's issue, then you can set the input file path on line 15 of `streaming/test_vocos` and then run `uv run python -m streaming.test_vocos` from the project root. This will output a `'streaming/vocoder_test_vocos.wav` file which contains the result of directly running the vocoder on the input (and skipping the model inference). 