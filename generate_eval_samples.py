"""
Generate eval samples by calling the running inference server (Server.ipynb).

Start the server first (run Cell 3 in Server.ipynb), then:
    python generate_eval_samples.py
    python generate_eval_samples.py --server http://localhost:8000
    python generate_eval_samples.py --text "One custom sentence."
"""

import argparse
import os
import sys
import requests

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "eval_samples")
SERVER_URL = "http://localhost:8000"

# Subset of the actual training sentences from setup.js.
# Using these means (a) WER ground truth is exact, (b) F0/MFCC comparison is
# same-phoneme-content rather than cross-utterance, (c) you can align against
# the corresponding segment in voice_training.wav if you later add forced alignment.
# Selected to cover: long/short, descriptive, numbers, questions, emotional, instructional.
EVAL_SENTENCES = [
    # Short — tests minimal-context voice conversion
    "Wait.",
    "Really?",
    # Numbers — edge case for RVC (complex phoneme sequences)
    "Forty-two ships had wrecked on those rocks — forty-two, and not one more, since the light was built.",
    "It takes approximately eight minutes and twenty seconds for sunlight to travel from the sun to the earth.",
    # Question — rising intonation contour
    "Have you ever watched the sea at night and wondered what it remembers?",
    # Reflective — varied rhythm, em-dashes
    "There is a kind of courage that looks exactly like standing still.",
    # Descriptive — long, sustained voice quality
    "The old lighthouse stood at the edge of the world, its beam sweeping the fog in slow, patient arcs.",
    # Instructional — mid-length, factual prosody
    "Hold the pen lightly — tension travels up through the hand and into the wrist faster than you expect.",
    # Emotional — tests expressiveness
    "She had expected grief to arrive loudly, but it came in small, quiet moments — a mug, a song, a name.",
    # Lyrical — tests rhythm and pacing
    "Rain in autumn sounds different from rain in spring — one is letting go, the other is arriving.",
]


def synth(text: str, server: str, timeout: int = 60) -> bytes:
    r = requests.post(f"{server}/synth", json={"text": text}, timeout=timeout)
    r.raise_for_status()
    return r.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=SERVER_URL)
    parser.add_argument("--text",   default=None, help="Single custom sentence")
    args = parser.parse_args()

    # Check server is up
    try:
        ping = requests.get(f"{args.server}/ping", timeout=5)
        ping.raise_for_status()
    except Exception as e:
        sys.exit(f"Server not reachable at {args.server}: {e}\nStart Server.ipynb Cell 3 first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sentences = [args.text] if args.text else EVAL_SENTENCES

    print(f"Server: {args.server}")
    print(f"Generating {len(sentences)} sample(s) → {OUTPUT_DIR}/\n")

    for i, text in enumerate(sentences):
        out_path = os.path.join(OUTPUT_DIR, f"sample_{i:02d}_rvc.wav")
        wav = synth(text, args.server)
        with open(out_path, "wb") as f:
            f.write(wav)
        print(f"  [{i:02d}] {len(wav):>8,} bytes  {out_path}")

    print(f"\nDone. Run:  python evaluate_metrics.py")


if __name__ == "__main__":
    main()
