# RVC Voice Cloning — Training & Inference

Local pipeline for training a personal voice-clone model with **RVC (Retrieval-based Voice Conversion)** via [Applio](https://github.com/IAHispano/Applio) v3.6.2, and serving text-to-speech through that cloned voice over a local HTTP API.

## Pipeline

```
TRAINING (Training.ipynb, run once)
  voice_training.wav
    → ffmpeg loudnorm + resample (40kHz mono)
    → Applio preprocess   — silence-aware slicing, denoise, normalize
    → Applio extract      — RMVPE pitch (F0) + ContentVec/HuBERT content embeddings
    → Applio index        — FAISS IVF index over embeddings
    → Applio train        — VITS-style GAN, 150 epochs
    → models/user_voice.pth + user_voice.index

INFERENCE (Server.ipynb, per request)
  text → edge-tts → MP3 → WAV (40kHz mono) → Applio VoiceConverter → cloned-voice WAV
    served via FastAPI on localhost:8000

EVALUATION (generate_eval_samples.py + evaluate_metrics.py)
  fixed test sentences → /synth → speaker similarity, pitch, timbre, WER, naturalness
```

## Requirements

- NVIDIA GPU (CUDA 12.x) — CPU training is impractically slow
- Python 3.12 venv with Applio's `requirements.txt` (PyTorch cu128, faiss-cpu, librosa, gradio, etc.)
- `ffmpeg` / `ffprobe`
- Jupyter kernel registered against the venv (`vclone_venv`)

## Setup

### 1. Create the virtual environment

Uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible installs (any `venv`/`pip` workflow works too).

```bash
# Install uv, if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a Python 3.12 venv next to this project
cd /home/zero/Desktop/Explore/VoiceCloning
uv venv --python 3.12 vclone_venv
```

### 2. Register it as a Jupyter kernel

```bash
uv pip install --python vclone_venv/bin/python ipykernel
vclone_venv/bin/python -m ipykernel install --user --name vclone_venv --display-name "vclone_venv"
```

Select the **vclone_venv** kernel in `Training.ipynb` / `Server.ipynb` before running any cells.

### 3. Install dependencies

`Training.ipynb` Cell 1 does this automatically the first time it's run — it clones Applio 3.6.2 into `./Applio`, installs its `requirements.txt` (PyTorch + CUDA, faiss-cpu, librosa, gradio, etc.) into `vclone_venv` via `uv`, then downloads the pretrained RMVPE / ContentVec / HiFi-GAN weights. No manual step is required for a normal run.

To install manually instead (e.g. Applio is already cloned, or you want the env ready before opening Jupyter):

```bash
# Core Applio dependencies (CUDA 12.x wheels)
uv pip install --python vclone_venv/bin/python \
  -r Applio/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match

# Extras used by Training.ipynb (plotting / monitoring)
uv pip install --python vclone_venv/bin/python \
  pydub matplotlib tensorboard scipy

# Extras used by Server.ipynb (TTS + API server)
uv pip install --python vclone_venv/bin/python \
  edge-tts fastapi "uvicorn[standard]" pydub nest_asyncio

# Pretrained model weights (RMVPE, ContentVec, HiFi-GAN)
vclone_venv/bin/python Applio/core.py prerequisites --models True --pretraineds_hifigan True
```

> Requires an NVIDIA GPU with CUDA 12.x drivers. On Linux you may also need `sudo apt-get install portaudio19-dev` (a `pyaudio` build dependency pulled in by Applio).

### 4. Prepare the Training Dataset (`voice_training.wav`)

To train a high-quality RVC model, you need a single, clean audio recording of the target voice (at least 5 to 10 minutes of continuous speech is recommended). 

You can generate this dataset by reading and recording the balanced sentences listed in [training_sentences.md](training_sentences.md):

1. **Environment**: Record in a quiet room with minimal background noise and echo.
2. **Recording**: Read the sentences at a natural, conversational pace. Try to capture a natural range of emotions, questions, and emphasis.
3. **Exporting**: Save or export the complete recorded audio as `voice_training.wav` directly in the root of this project folder.
   - *Technical Specification*: Export as mono, 16-bit or 24-bit PCM WAV at 44.1 kHz or 48 kHz. The preprocessing notebook will automatically handle downmixing, volume normalization, and resampling to 40 kHz.


## Usage

### 1. Train a voice model

Run `Training.ipynb` top to bottom (kernel: `vclone_venv`). Produces:

```
models/user_voice.pth      # trained generator weights
models/user_voice.index    # FAISS retrieval index
training_plots.png         # loss curves, checkpoint sizes, input spectrogram
```

Training runs `core.py preprocess → extract → index → train` as subprocesses and renders a live 4-panel loss dashboard (gen/disc loss, mel loss, sub-losses, LR schedule) refreshed every 30s. An overtraining detector auto-stops if loss plateaus for 50 epochs.

### 2. Serve inference

Run `Server.ipynb` (requires a trained model). Starts a FastAPI server:

```bash
curl http://localhost:8000/ping
curl -X POST http://localhost:8000/synth \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world"}' -o out.wav
```

`/synth` pipeline: Edge-TTS (`en-US-JennyNeural`) → resample → RVC conversion (`f0_method=rmvpe`, `index_rate=0.75`, `protect=0.5`) → WAV response.

### 3. Evaluate quality

```bash
python generate_eval_samples.py          # synthesizes 10 benchmark sentences via the running server
python evaluate_metrics.py               # scores them against voice_training.wav
```

Metrics and targets:

| Metric | Method | Target |
|---|---|---|
| Speaker similarity | Resemblyzer GE2E cosine similarity | ≥ 0.80 |
| Pitch offset | Median F0 (semitones) vs. reference | \|offset\| < 1.0 st |
| Timbre distance | MFCC cosine distance | < 0.10 |
| Intelligibility | Whisper WER on output audio | < 5% |
| Naturalness | RMS level / silence ratio | compare vs. TTS baseline |

Full results are written to `eval_report.json`.

## Project Structure

```
rvc-training-inference/
├── Training.ipynb           # one-time: preprocess → extract → index → train → save model
├── Server.ipynb             # TTS + RVC inference server (localhost:8000)
├── generate_eval_samples.py # synthesize fixed benchmark sentences via the server
├── evaluate_metrics.py      # score synthesized samples against the reference voice
├── voice_training.wav       # source recording (training data)
├── models/                  # user_voice.pth + user_voice.index (created by training)
├── dataset/                 # preprocessed training audio (created by training)
├── eval_samples/            # generated benchmark WAVs (created by eval scripts)
└── Applio/                  # vendored RVC engine (v3.6.2)
```

## RVC Engine (Applio)

The cloning quality comes from Applio's `rvc/` package:

- **Model** — a VITS-style conditional VAE (content encoder + posterior encoder + normalizing flow) with a **Neural Source-Filter (NSF)** HiFi-GAN decoder: pitch is synthesized as an explicit harmonic-sine excitation and fused into the vocoder at every upsampling stage, giving accurate, stable pitch reproduction. Trained adversarially against a multi-period/multi-scale discriminator (LSGAN + feature-matching + mel L1 + KL loss).
- **Content embeddings** — ContentVec (a speaker-disentangled HuBERT variant) extracts frame-level linguistic features, decoupling *what is said* from *who says it*; no duration predictor is needed since embeddings are already frame-aligned.
- **Pitch extraction** — RMVPE (Deep U-Net + BiGRU pitch-salience classifier), the default used here; FCPE and CREPE are also available.
- **Retrieval (the "R" in RVC)** — a FAISS index over the target voice's training embeddings; at inference, each source content frame is blended with its `k=8` nearest neighbors in that index (`index_rate`), pulling the output toward the trained voice's actual timbre manifold.

See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for a full breakdown of the architecture, training loop, and inference pipeline.

## Key Configuration

| Parameter | Value | Notes |
|---|---|---|
| `SAMPLE_RATE` | 40000 | Training/model sample rate |
| `EPOCHS` | 150 | ~15–30 min on an RTX 3090 |
| `f0_method` | `rmvpe` | Pitch estimator |
| `embedder_model` | `contentvec` | Content embedding model |
| `index_rate` | 0.75 | Retrieval blend strength at inference |
| `protect` | 0.5 | Consonant protection (0.5 = off) |
| `pitch` | 0 | No key transpose |

## Notes

- Training is single-speaker and must be re-run per voice; there is no zero-shot cloning.
- The server subprocess isolates its Python environment from the notebook kernel to avoid numpy/scipy version conflicts.
- `Applio/` is a vendored upstream clone — see its own `README.md`/`LICENSE` (MIT) for terms of use.
