"""
Evaluate voice cloning quality of RVC outputs against the reference voice.

Metrics
-------
1. Speaker similarity  — resemblyzer cosine similarity (reference WAV vs each RVC output)
   Target: > 0.80  (0.75 = marginal, 0.85+ = good clone)

2. F0 statistics       — pitch median and spread compared to reference distribution
   Target: median within ~1 semitone; std within 50% of reference

3. MFCC distance       — mean MFCC vector cosine distance (timbre proxy)
   Target: < 0.10  (lower = more similar timbre)

4. Intelligibility     — Whisper WER on RVC output vs original text
   Target: < 5%  (RVC should not garble words)

5. Naturalness         — signal RMS and silence ratio (proxy for artifact presence)
   Target: no hard threshold — compare RVC vs TTS baseline

Usage:
    python evaluate_metrics.py                     # scores eval_samples/sample_*_rvc.wav
    python evaluate_metrics.py --ref path/to.wav   # custom reference
    python evaluate_metrics.py --no-whisper        # skip ASR (faster)
"""

import argparse
import glob
import json
import os
import sys
import warnings

import librosa
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "eval_samples")
REFERENCE   = os.path.join(BASE_DIR, "voice_training.wav")
REPORT_PATH = os.path.join(BASE_DIR, "eval_report.json")

# ─── helpers ──────────────────────────────────────────────────────────────────

def load_mono(path: str, sr: int = 16000) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def extract_f0(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Return voiced F0 values in Hz (non-zero only)."""
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=600, sr=sr,
                             frame_length=2048, hop_length=512)
    return f0[~np.isnan(f0) & (f0 > 0)]


def mfcc_mean(y: np.ndarray, sr: int = 16000, n_mfcc: int = 20) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def hz_to_semitones(f: float) -> float:
    return 12.0 * np.log2(f / 440.0)


def semitone_distance(f_ref: float, f_out: float) -> float:
    """Signed semitone difference (positive = output is higher)."""
    if f_ref <= 0 or f_out <= 0:
        return float("nan")
    return hz_to_semitones(f_out) - hz_to_semitones(f_ref)


def silence_ratio(y: np.ndarray, top_db: int = 30) -> float:
    """Fraction of frames that are silent."""
    intervals = librosa.effects.split(y, top_db=top_db)
    voiced_samples = sum(b - a for a, b in intervals)
    return 1.0 - voiced_samples / max(len(y), 1)


# ─── speaker similarity ───────────────────────────────────────────────────────

def speaker_similarity(ref_path: str, out_path: str) -> float:
    """
    Cosine similarity between GE2E speaker embeddings.
    Returns value in [-1, 1]; typically 0.6–0.95 for same speaker.
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        print("  [skip] resemblyzer not installed")
        return float("nan")

    encoder = VoiceEncoder()
    ref_wav = preprocess_wav(ref_path)
    out_wav = preprocess_wav(out_path)

    ref_emb = encoder.embed_utterance(ref_wav)
    out_emb = encoder.embed_utterance(out_wav)
    return cosine_sim(ref_emb, out_emb)


# ─── ASR intelligibility ──────────────────────────────────────────────────────

_whisper_model = None

def transcribe(path: str) -> str:
    global _whisper_model
    try:
        import whisper
        if _whisper_model is None:
            _whisper_model = whisper.load_model("base.en")
        result = _whisper_model.transcribe(path, language="en")
        return result["text"].strip()
    except ImportError:
        return ""
    except Exception as e:
        return f"[whisper error: {e}]"


def wer(reference_text: str, hypothesis: str) -> float:
    try:
        from jiwer import wer as _wer
        return _wer(reference_text.lower(), hypothesis.lower())
    except ImportError:
        return float("nan")


# ─── per-file scoring ─────────────────────────────────────────────────────────

def score_file(rvc_path: str, ref_f0: np.ndarray,
               ref_mfcc: np.ndarray, expected_text: str,
               use_whisper: bool = True) -> dict:
    out_y = load_mono(rvc_path)

    # F0
    out_f0 = extract_f0(out_y)
    ref_med  = float(np.median(ref_f0))  if len(ref_f0)  else 0.0
    out_med  = float(np.median(out_f0))  if len(out_f0)  else 0.0
    out_std  = float(np.std(out_f0))     if len(out_f0)  else 0.0
    pitch_offset_st = semitone_distance(ref_med, out_med)

    # MFCC timbre
    out_mfcc  = mfcc_mean(out_y)
    mfcc_dist = 1.0 - cosine_sim(ref_mfcc, out_mfcc)  # lower = closer

    # Naturalness proxies
    sil_ratio = silence_ratio(out_y)
    rms_db    = float(20 * np.log10(np.sqrt(np.mean(out_y ** 2)) + 1e-9))

    result = {
        "file":            os.path.basename(rvc_path),
        "duration_s":      round(len(out_y) / 16000, 2),
        "f0_median_hz":    round(out_med, 1),
        "f0_std_hz":       round(out_std, 1),
        "pitch_offset_st": round(pitch_offset_st, 2),
        "mfcc_distance":   round(mfcc_dist, 4),
        "silence_ratio":   round(sil_ratio, 3),
        "rms_db":          round(rms_db, 1),
    }

    if use_whisper and expected_text:
        hyp = transcribe(rvc_path)
        result["transcription"] = hyp
        result["wer"] = round(wer(expected_text, hyp), 3) if hyp else float("nan")

    return result


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",        default=REFERENCE, help="Reference WAV (voice_training.wav)")
    parser.add_argument("--samples",    default=SAMPLES_DIR, help="Directory with *_rvc.wav files")
    parser.add_argument("--no-whisper", action="store_true")
    parser.add_argument("--sentences",  default=None, help="JSON file with sentence list")
    args = parser.parse_args()

    if not os.path.exists(args.ref):
        sys.exit(f"Reference not found: {args.ref}")

    rvc_files = sorted(glob.glob(os.path.join(args.samples, "*_rvc.wav")))
    if not rvc_files:
        sys.exit(f"No *_rvc.wav files found in {args.samples}\nRun infer_samples.py first.")

    # Load sentences to match files → expected text
    if args.sentences:
        sentences = json.load(open(args.sentences))
    else:
        from generate_eval_samples import EVAL_SENTENCES as sentences

    print(f"Reference: {args.ref}")
    print(f"Samples:   {len(rvc_files)} files\n")

    # ── precompute reference features once ───────────────────────────────────
    print("Computing reference features...")
    ref_y   = load_mono(args.ref, sr=16000)
    # Use a random 60 s window from the reference (full file is 13 min)
    clip_samples = 16000 * 60
    start = max(0, len(ref_y) // 4)             # skip first few minutes (warmup)
    ref_clip = ref_y[start : start + clip_samples]
    ref_f0   = extract_f0(ref_clip)
    ref_mfcc = mfcc_mean(ref_clip)
    ref_med  = float(np.median(ref_f0)) if len(ref_f0) else 0.0
    ref_std  = float(np.std(ref_f0))    if len(ref_f0) else 0.0
    print(f"  F0 median: {ref_med:.1f} Hz  std: {ref_std:.1f} Hz")

    # ── speaker similarity (whole reference vs each output) ───────────────────
    print("\nSpeaker similarity (GE2E cosine):")
    sim_scores = []
    for path in rvc_files:
        sim = speaker_similarity(args.ref, path)
        sim_scores.append(sim)
        bar = "█" * int(sim * 20) if not np.isnan(sim) else "n/a"
        print(f"  {os.path.basename(path):35s}  {sim:.3f}  {bar}")

    mean_sim = float(np.nanmean(sim_scores))
    print(f"\n  Mean similarity: {mean_sim:.3f}  ", end="")
    if mean_sim >= 0.85:   print("✓ GOOD (≥0.85)")
    elif mean_sim >= 0.75: print("~ OK   (0.75–0.85, could improve)")
    else:                  print("✗ POOR (<0.75 — check pitch shift, index_rate, epochs)")

    # ── per-file metrics ──────────────────────────────────────────────────────
    print("\nPer-file metrics:")
    print(f"  {'file':35s} {'dur':>5} {'F0_med':>7} {'Δst':>5} {'MFCC_d':>7} {'sil':>5} {'RMS':>6} {'WER':>6}")
    print("  " + "-" * 85)

    all_results = []
    for i, path in enumerate(rvc_files):
        text = sentences[i] if i < len(sentences) else ""
        r = score_file(path, ref_f0, ref_mfcc, text,
                       use_whisper=not args.no_whisper)
        r["speaker_similarity"] = round(sim_scores[i], 4)
        all_results.append(r)

        wer_s = f"{r.get('wer', float('nan')):.2f}" if "wer" in r else "  n/a"
        print(
            f"  {r['file']:35s} "
            f"{r['duration_s']:>5.1f} "
            f"{r['f0_median_hz']:>7.1f} "
            f"{r['pitch_offset_st']:>+5.2f} "
            f"{r['mfcc_distance']:>7.4f} "
            f"{r['silence_ratio']:>5.3f} "
            f"{r['rms_db']:>6.1f} "
            f"{wer_s:>6}"
        )

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────")
    print(f"  Speaker similarity:    {mean_sim:.3f}  (target ≥ 0.80)")
    pitch_offsets = [r["pitch_offset_st"] for r in all_results if not np.isnan(r["pitch_offset_st"])]
    if pitch_offsets:
        mean_offset = np.mean(pitch_offsets)
        print(f"  Pitch offset (mean):   {mean_offset:+.2f} st  (target |offset| < 1.0 st)")
        if abs(mean_offset) > 1.5:
            recommend = int(round(-mean_offset))
            print(f"  !! Try --pitch {recommend:+d} in infer_samples.py to correct pitch")
    mfcc_dists = [r["mfcc_distance"] for r in all_results]
    print(f"  MFCC distance (mean):  {np.mean(mfcc_dists):.4f}  (target < 0.10)")
    if "wer" in all_results[0]:
        wers = [r["wer"] for r in all_results if not np.isnan(r.get("wer", float("nan")))]
        if wers:
            print(f"  WER (mean):            {np.mean(wers)*100:.1f}%  (target < 5%)")

    # ── save full report ──────────────────────────────────────────────────────
    report = {
        "reference": args.ref,
        "ref_f0_median_hz": round(ref_med, 1),
        "ref_f0_std_hz":    round(ref_std, 1),
        "mean_speaker_similarity": round(mean_sim, 4),
        "samples": all_results,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
