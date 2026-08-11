# RVC Training & Inference — Technical Report

Analysis of `rvc-training-inference/`: a local voice-cloning pipeline built on **Applio v3.6.2** (a maintained fork/toolkit around **RVC — Retrieval-based Voice Conversion**). This folder trains a personal voice model from a single WAV recording, then serves TTS-plus-voice-conversion over a local FastAPI endpoint, with a standalone objective evaluation harness.

---

## 1. File Inventory

| File | Role |
|---|---|
| `Training.ipynb` | One-time pipeline: clone/install Applio → preprocess `voice_training.wav` → extract F0 + speaker embeddings → build FAISS index → train GAN model → save `.pth`/`.index` |
| `Server.ipynb` | Long-running FastAPI server: Edge-TTS text→speech → RVC voice conversion → returns WAV |
| `generate_eval_samples.py` | CLI: calls the running server on 10 fixed benchmark sentences, saves outputs to `eval_samples/` |
| `evaluate_metrics.py` | CLI: scores those outputs against the reference voice (speaker similarity, pitch, timbre, intelligibility, naturalness) |
| `voice_training.wav` | ~13 min source recording used as the sole training dataset |
| `Applio/` | Vendored clone of [IAHispano/Applio](https://github.com/IAHispano/Applio) `3.6.2` — the actual RVC engine (model code, training loop, inference pipeline, Gradio UI, CLI) |

Not covered in depth (present in `Applio/` but unused by this project): `rvc/realtime/` (streaming voice-changer client) and `tabs/` (Gradio web UI) — both are alternate front-ends over the same `rvc/` engine and `core.py` CLI that this project drives directly/via subprocess instead.

---

## 2. End-to-End Flow

```
TRAINING (Training.ipynb, run once)
  voice_training.wav
    → ffmpeg: loudnorm + resample to 40kHz mono
    → Applio core.py preprocess   (slice into ~3s chunks, denoise, normalize)
    → Applio core.py extract      (RMVPE F0  +  ContentVec/HuBERT embeddings, per chunk)
    → Applio core.py index        (cluster + FAISS IVF index over embeddings)
    → Applio core.py train        (VITS-style GAN, 150 epochs)
    → models/user_voice.pth + user_voice.index

INFERENCE (Server.ipynb, per request)
  text
    → edge-tts (en-US-JennyNeural)        → MP3
    → pydub                               → 40kHz mono WAV
    → Applio VoiceConverter.convert_audio → RVC-converted WAV (target voice)
    → HTTP response

EVALUATION (generate_eval_samples.py + evaluate_metrics.py)
  10 fixed sentences → POST /synth → eval_samples/*.wav
    → resemblyzer speaker similarity, F0 offset, MFCC distance, Whisper WER, RMS/silence
    → eval_report.json
```

---

## 3. Top-Level Orchestration

### 3.1 `Training.ipynb`
7 cells, run once per voice:

1. **Install** — clones Applio `3.6.2` if absent, installs `portaudio19-dev`, installs Applio's `requirements.txt` via `uv` (CUDA 12.4 wheel index), plus `pydub`/`matplotlib`/`tensorboard`/`scipy`, then downloads pretrained models (RMVPE, ContentVec, HiFi-GAN) via `core.py prerequisites`.
2. **Config** — `MODEL_NAME='user_voice'`, `SAMPLE_RATE=40000`, `EPOCHS=150`; checks CUDA availability.
3. **Prepare audio** — `ffmpeg` loudness-normalizes (`loudnorm=I=-16:TP=-1.5:LRA=11`) and resamples the source WAV to 40kHz mono; warns if under 120s.
4. **Preprocess → Extract → Index** — three sequential `core.py` subprocess calls:
   - `preprocess --cut_preprocess Automatic` (silence-based slicing)
   - `extract --f0_method rmvpe --embedder_model contentvec` (pitch + content embeddings)
   - `index --index_algorithm Auto` (FAISS index build)
5. **Train with live monitoring** — launches `core.py train` as a subprocess (`batch_size=8`, `vocoder=HiFi-GAN`, `overtraining_detector=True`, `overtraining_threshold=50`, checkpoint every 10 epochs) on a background thread; the notebook's main thread polls stdout (regex-parses `epoch=`/`smoothed_loss_gen=`/`smoothed_loss_disc=`) and TensorBoard event files every 30s, redrawing a 4-panel live matplotlib dashboard (gen/disc loss, mel loss, sub-losses [feature-match/KL/adversarial], learning-rate schedule).
6. **Save model** — copies the final (or highest-epoch checkpoint) `.pth` and its `.index` from Applio's `logs/user_voice/` into this project's `models/user_voice.pth`/`.index`.
7. **Diagnostics** (re-runnable) — pulls all TensorBoard scalar tags found under the log dir, plots loss curves, checkpoint-size bar chart, and a 30s spectrogram of the training audio; saves `training_plots.png`.

### 3.2 `Server.ipynb`
3 functional cells:

1. **Install** — adds `edge-tts`, `fastapi`, `uvicorn`, `pydub`, `nest_asyncio` on top of Applio's deps.
2. **Config** — asserts `models/user_voice.pth`/`.index` exist (i.e., Training.ipynb must run first).
3. **Run server** — writes a self-contained server script to `/tmp/rvc_local_server.py` and launches it as a **subprocess** (isolates the server's numpy/scipy/torch environment from the notebook kernel). The script:
   - Instantiates Applio's `VoiceConverter` once at startup.
   - Runs a synchronous warmup inference ("Hello.") to force CUDA kernel compilation before serving real traffic.
   - Exposes a FastAPI app with CORS-open `/ping` and `POST /synth`.
   - `/synth` pipeline: `edge_tts.Communicate(text, 'en-US-JennyNeural')` → MP3 bytes → `pydub` converts to 40kHz mono WAV → `VoiceConverter.convert_audio(pitch=0, f0_method='rmvpe', index_rate=0.75, volume_envelope=1.0, protect=0.5)` → returns WAV bytes. Both the WAV conversion and the RVC conversion run on a single-worker `ThreadPoolExecutor` to serialize GPU access.
   - The notebook cell streams the subprocess's stdout back into the notebook and blocks until interrupted.
4. **Smoke tests** — a separate cell hits `/ping` and `/synth` with `requests`, asserting a valid WAV (`RIFF` header) is returned, and saves a sample to `test_output.wav`.

### 3.3 `generate_eval_samples.py`
Standalone script (no Applio import — talks to the already-running server over HTTP). Defines 10 hand-picked benchmark sentences chosen to stress different conditions: minimal-context (short utterances), numeric/complex phonemes, question intonation, reflective/em-dash rhythm, long descriptive passages, instructional prosody, emotional expressiveness, and lyrical rhythm. Posts each to `/synth`, saves `eval_samples/sample_NN_rvc.wav`. Sentences are drawn from the same pool used for voice-training prompts elsewhere in the broader project (per project memory, `setup.js`), so WER ground truth and F0/MFCC comparisons stay same-phoneme-content rather than cross-utterance.

### 3.4 `evaluate_metrics.py`
Objective voice-clone quality harness, independent of Applio (uses `librosa`, optional `resemblyzer`/`whisper`/`jiwer`). For each generated sample vs. a 60s window of the reference recording (skipping the first quarter to avoid mic-warmup artifacts):

- **Speaker similarity** — cosine similarity between `resemblyzer` GE2E speaker embeddings of reference vs. output. Target ≥ 0.80.
- **F0 statistics** — pitch median/std via `librosa.pyin`; reports pitch offset in semitones vs. reference, with an automatic `--pitch` correction suggestion if the mean offset exceeds 1.5 semitones.
- **MFCC distance** — `1 - cosine_sim` of mean 20-coefficient MFCC vectors, as a timbre proxy. Target < 0.10.
- **Intelligibility** — Whisper (`base.en`) transcribes the output; word error rate (`jiwer`) against the known input sentence. Target < 5%.
- **Naturalness proxies** — RMS level (dB) and silence ratio (`librosa.effects.split`), no hard threshold — meant for comparing RVC output against a TTS-only baseline.

Prints a formatted per-file table plus a summary, and writes `eval_report.json` with the full numeric results.

---

## 4. Applio Core Engine (`Applio/rvc/`)

Applio's `rvc/` package (~12,200 lines) implements the actual ML system. Architecturally it is a **VITS-derived conditional VAE + normalizing-flow + GAN vocoder**, adapted for voice conversion (frame-aligned content embeddings instead of text/phonemes, so no duration predictor or attention alignment is needed) and made **pitch-accurate** via a **Neural Source-Filter (NSF)** excitation signal, plus a **FAISS retrieval mechanism** that gives RVC its name and its main advantage over vanilla voice-conversion VAEs.

### 4.1 Model Architecture — `rvc/lib/algorithm/`

**`Synthesizer`** (`synthesizers.py`) is the generator, composed of four submodules, all conditioned on a learned per-speaker embedding `g` (`emb_g`, a simple `nn.Embedding` indexed by integer speaker ID — RVC is not zero-shot; each trained model is voice-specific):

| Submodule | File | Role |
|---|---|---|
| `TextEncoder` | `encoders.py` | Despite the name, consumes HuBERT/ContentVec content embeddings (not text). Linear-projects them, adds a discrete 256-bin pitch embedding, runs a 6-layer VITS-style Transformer (relative-position self-attention, `attentions.py`) → outputs the **prior** distribution `(m_p, logs_p)` over the latent. |
| `PosteriorEncoder` | `encoders.py` | Training-only. Takes the ground-truth linear spectrogram, runs a WaveNet-style dilated-conv stack conditioned on `g`, outputs the **posterior** `(m_q, logs_q)` and a reparameterized sample `z`. Deleted at inference time. |
| `ResidualCouplingBlock` (flow) | `residuals.py` | 4 affine/mean-only coupling layers + permutation flips, mapping between posterior-space `z` and prior-space `z_p`. Zero-initialized final layer for training stability. |
| Vocoder/decoder (one of 4) | `generators/*.py` | Converts the sampled latent + pitch curve into a waveform. |

**Vocoder options** (selected by `vocoder` config string):
- **`HiFiGANGenerator`** — classic HiFi-GAN v1 (transposed-conv upsampling + multi-kernel residual blocks), no pitch input.
- **`HiFiGANNSFGenerator`** — the default RVC decoder. Adds an NSF harmonic-sine excitation signal synthesized directly from the frame-level F0 curve (voiced/unvoiced masking, cumulative-phase synthesis to avoid inter-frame phase jumps), injected into the generator at every upsampling resolution — this is what gives RVC accurate, stable pitch reproduction.
- **`RefineGANGenerator`** — heavier, higher-fidelity alternative: parallel-downsampled sine source fused at each decoder stage with anti-aliased resampling, `ParallelResBlock`s (kernels 3/7/11 averaged, AdaIN-style noise injection). Paired automatically with the `v3` multi-resolution-STFT discriminator and multi-scale mel loss.
- **`HiFiGANMRFGenerator`** — multi-receptive-field variant, wired the same way.

**Discriminator** — `MultiPeriodDiscriminator` (`discriminators.py`) ensembles a multi-scale waveform discriminator (`DiscriminatorS`) with several multi-period 2D discriminators (`DiscriminatorP`, periods `[2,3,5,7,11,17(,23,37)]`, classic HiFi-GAN MPD design), plus optional multi-resolution STFT discriminators (`DiscriminatorR`) when using the `v3`/RefineGAN configuration.

**Training vs. inference forward pass**: training runs the full VAE (posterior → flow → decode a random segment, for reconstruction/adversarial loss against real audio); inference samples directly from the content-encoder's **prior** (with a `0.66666` temperature-reduction constant to reduce artifact variance), pushes it through the flow in reverse, and decodes — no reference audio needed at inference, only the target text-independent content signal.

### 4.2 Pitch (F0) Extraction — `rvc/lib/predictors/`

Three interchangeable pitch estimators, dispatched by a `method` string in both the training extractor and the inference pipeline:

- **RMVPE** (default) — Deep U-Net over a 128-band log-mel spectrogram + bidirectional GRU → 360-bin pitch-salience classification, decoded to continuous Hz via local weighted averaging. Robust to noisy/polyphonic input.
- **FCPE** — Conformer-style estimator (`torchfcpe`) using depthwise-conv + linear/Performer attention for efficient long-sequence pitch salience. Lighter/faster alternative.
- **CREPE / CREPE-tiny** — the standard CNN pitch tracker (`torchcrepe`), with periodicity-based post-filtering.
- (Older WORLD-based `harvest`/`dio`/`pm` methods from upstream RVC are **not present** in this Applio build.)

F0 is used two ways downstream: quantized into a 256-bin coarse pitch class feeding the `TextEncoder`'s discrete pitch embedding, and kept as a continuous Hz curve (`pitchf`) feeding the NSF decoder's excitation generator directly. Optional autotune (snap-to-nearest-semitone) and an automatic median-pitch key-shift heuristic are also implemented in the inference pipeline.

### 4.3 Content/Speaker Embedding — ContentVec/HuBERT

`load_embedding()` (`rvc/lib/utils.py`) loads one of several HuBERT-family models (default: **ContentVec**, a HuBERT variant trained with a speaker-disentanglement objective so its output embeddings carry linguistic/content information with reduced speaker identity — exactly what a content-preserving voice converter needs). Applio wraps HuggingFace's `HubertModel` to restore the `final_proj` layer needed to match the original fairseq ContentVec checkpoint.

Both training extraction and inference run the embedder on 16kHz audio and take `last_hidden_state` (768-dim for v2 models) as the per-frame content vector. Because HuBERT's output frame rate (~50Hz) is roughly half the acoustic/pitch frame rate used elsewhere in the pipeline, embeddings are **frame-repeated/interpolated 2×** before being fed into the synthesizer, both at training and inference time.

### 4.4 FAISS Retrieval — the "R" in RVC

This is Applio/RVC's core differentiator from a plain VAE voice converter. After training-set embeddings are extracted, `extract_index.py` concatenates all per-utterance content vectors, optionally `MiniBatchKMeans`-compresses very large sets to 10k centroids, and builds a **FAISS `IVF{n},Flat` index** over them.

At **inference time**, for each frame of the source audio's content embedding, the pipeline searches the target speaker's index for its `k=8` nearest neighbors, inverse-square-distance-weights and averages them into a "retrieved" feature vector guaranteed to be in-distribution for the target voice's timbre — then linearly blends it with the original embedding by `index_rate` (`retrieved*index_rate + original*(1-index_rate)`). This nudges potentially source-speaker-leaky content features toward the trained voice's actual data manifold without any additional training, at the cost of a per-frame nearest-neighbor search.

### 4.5 Training Pipeline — `rvc/train/`

Four sequential stages, each invoked as a `core.py` subcommand (matching what `Training.ipynb` Cell 4/5 calls):

1. **Preprocess** (`preprocess/preprocess.py`) — per-file high-pass filter (48Hz), RMS loudness normalization, optional denoising, then silence-aware slicing (`Slicer`, RMS-threshold silence detection) into ~3s chunks with 0.3s overlap. Each chunk is saved both at training sample rate and resampled to 16kHz (for F0/embedding extraction).
2. **Extract** (`extract/extract.py`) — parallelized (multi-process/multi-GPU) F0 extraction and embedding extraction per chunk; writes coarse+fine pitch `.npy` and content-embedding `.npy` files, then generates `config.json` and a pipe-delimited `filelist.txt` linking each utterance's audio/features/pitch/speaker-id.
3. **Index** (`process/extract_index.py`) — builds the FAISS retrieval index described in §4.4.
4. **Train** (`train.py`) — the GAN training loop:
   - Mixed precision (`bf16`/`fp16`/`fp32`, configurable) via `torch.amp`; multi-GPU via `DistributedDataParallel`.
   - Length-bucketed batch sampling (`DistributedBucketSampler`) to minimize padding waste, standard VITS-family trick.
   - `AdamW` optimizers (β=(0.8, 0.99)) with independent exponential LR decay (γ=0.999875) for generator and discriminator; resumes from checkpoints or Applio's pretrained warm-start weights.
   - **Loss function** (mirrors VITS exactly): discriminator LSGAN loss; generator = adversarial loss + feature-matching L1 loss (over discriminator intermediate activations) + **mel-spectrogram L1 reconstruction loss** (or multi-resolution mel loss for RefineGAN) + **KL divergence** between the flow-mapped posterior and the content-encoder prior.
   - **Overtraining detection** — an Applio-specific addition: tracks an exponential moving average of gen/disc loss across epochs and auto-stops training if it plateaus/worsens for `overtraining_threshold` consecutive epochs (this project sets `overtraining_threshold=50`), keeping the best checkpoint.
   - Checkpoints (`G_*.pth`/`D_*.pth`) saved every `save_every_epoch` (10, in this project); a slim inference-only `.pth` is exported at milestones.
   - Training automatically chains into index-building on completion.

### 4.6 Inference Pipeline — `rvc/infer/`

`VoiceConverter.convert_audio()` (`infer.py`) is the entry point used by `Server.ipynb`:

1. Loads/resamples input audio to 16kHz; optionally splits long audio into silence-based chunks for parallelizable processing.
2. Delegates each chunk to `Pipeline.pipeline()` (`pipeline.py`):
   - High-pass filter, then optional low-energy-point windowing for very long inputs.
   - **F0 estimation** via the selected method (§4.2), with semitone key-shift (`pitch`), optional autotune.
   - **Content embedding** extraction via the HuBERT/ContentVec model.
   - **`protect` mechanism**: for unvoiced-adjacent frames, blends retrieved features back toward the original (non-retrieved) embedding by `(1-protect)`, preventing the FAISS retrieval step from overwriting breathy/consonant timbre. `protect=0.5` (used by this project's server) effectively disables this.
   - **FAISS retrieval blending** at the configured `index_rate` (§4.4) — this project uses `0.75`.
   - 2× frame-rate upsampling to match acoustic granularity, then `Synthesizer.infer()` — prior sampling → reverse flow → NSF/HiFi-GAN decode (§4.1).
   - Windows are edge-trimmed and reassembled.
3. **`volume_envelope`** (this project: `1.0`) optionally blends the converted audio's RMS loudness contour with the source's, to preserve original dynamics/expression rather than let the vocoder's flatter envelope dominate.
4. Optional post-processing: noise reduction, Pedalboard effects chain (reverb, limiter, compressor, etc. — unused by default here), format conversion, clipping-safety normalization.

### 4.7 `core.py` — CLI Orchestration

A 2,420-line `argparse` CLI that is almost entirely a thin shim: each subcommand (`preprocess`, `extract`, `index`, `train`, `infer`, `batch_infer`, `tts`, `model_information`, `model_blender`, `tensorboard`, `download`, `prerequisites`, `audio_analyzer`) assembles arguments and either calls into the corresponding `rvc/` module directly or subprocess-launches the corresponding script. This project's notebooks call `preprocess`/`extract`/`index`/`train`/`prerequisites` as subprocesses during training, and import `VoiceConverter` directly (bypassing `core.py`) for the server's inference path.

---

## 5. Key Parameters Used by This Project

| Parameter | Value | Effect |
|---|---|---|
| `SAMPLE_RATE` | 40000 | Training/model sample rate |
| `EPOCHS` | 150 | Total training epochs |
| `batch_size` | 8 | Training batch size |
| `f0_method` | `rmvpe` | Pitch estimator (both training extraction and inference) |
| `embedder_model` | `contentvec` | Content/speaker embedding model |
| `index_algorithm` | `Auto` | FAISS index build strategy (auto-selects KMeans compression if dataset is large) |
| `vocoder` | `HiFi-GAN` | NSF-conditioned HiFi-GAN decoder |
| `overtraining_detector` / `threshold` | `True` / `50` | Auto-stop if loss plateaus for 50 epochs |
| `pitch` | `0` | No key transpose at inference (same speaker gender assumed) |
| `index_rate` | `0.75` | Strong retrieval blending toward target voice |
| `protect` | `0.5` | Retrieval protection disabled (full retrieval blend applied everywhere) |
| `volume_envelope` | `1.0` | Converted audio's own loudness envelope kept as-is |

---

## 6. Summary of Techniques Implemented

- **VITS-style conditional VAE** (posterior encoder + prior content encoder + normalizing flow) adapted for voice conversion by replacing text input with frame-aligned self-supervised content embeddings (no duration predictor needed).
- **Neural Source-Filter (NSF) vocoding** — explicit harmonic-sine excitation from the F0 curve injected into a HiFi-GAN-style decoder, for pitch-accurate, artifact-resistant waveform synthesis (with a RefineGAN alternative for higher fidelity).
- **Self-supervised content embeddings** (ContentVec/HuBERT) as a speaker-disentangled linguistic representation, decoupling *what is said* from *who says it*.
- **FAISS approximate nearest-neighbor retrieval** blending source content embeddings toward the target speaker's actual training-data manifold at inference — the defining "retrieval-based" mechanism of RVC.
- **Adversarial (GAN) training** with a HiFi-GAN-style multi-period/multi-scale (optionally multi-resolution-STFT) discriminator ensemble, combined loss of LSGAN adversarial + feature-matching + mel-spectrogram L1 + KL divergence.
- **Engineering robustness features**: mixed-precision + multi-GPU training, length-bucketed batching, automatic overtraining detection/early stopping, silence-aware audio slicing, protect/index-rate/volume-envelope inference-time controls for quality tuning.
- **Orchestration layer** (this project's notebooks) wraps Applio's CLI/Python API into a fully local (no Colab/Drive/Firebase) train-once → serve-locally workflow, fronted by Edge-TTS for text input and a FastAPI server for the extension/client to call.
- **Objective evaluation harness** — independent of the RVC engine, using speaker-embedding cosine similarity, pitch/timbre distance metrics, and ASR-based intelligibility (WER) to quantify voice-clone quality against fixed targets.
