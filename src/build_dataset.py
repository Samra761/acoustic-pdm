"""
build_dataset.py
End-to-end: raw CSVs -> chunked audio -> Mel-spectrograms -> saved .npz arrays.

Run this once after the raw MAFAULDA data is unzipped into data/raw/.
Can build either the microphone dataset (default) or the vibration dataset,
so acoustic-only and vibration-only results can be compared later.

Output:
    data/processed/spectrograms_mic.npz        (channel="mic")
    data/processed/spectrograms_vibration.npz  (channel="vibration")
each containing:
    X       : (N, N_MELS, TARGET_TIME_FRAMES) float32 spectrograms
    y       : (N,) string labels
    groups  : (N,) source filepath, so we can do file-level (not chunk-level)
              train/test splitting and avoid leakage between overlapping windows
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

from data_loader import build_file_index, load_audio_signal
from feature_extraction import chunk_signal, signal_to_mel_spectrogram

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def main(channel="mic"):
    index = build_file_index()
    print(f"Found {len(index)} source files across {index['label'].nunique()} classes.")
    print(f"Building dataset using channel: {channel}")

    X, y, groups = [], [], []
    for _, row in tqdm(index.iterrows(), total=len(index), desc="Extracting features"):
        signal, sr = load_audio_signal(row["filepath"], channel=channel)
        chunks = chunk_signal(signal, sr)
        for chunk in chunks:
            spec = signal_to_mel_spectrogram(chunk, sr)
            X.append(spec)
            y.append(row["label"])
            groups.append(row["filepath"])

    X = np.stack(X)
    y = np.array(y)
    groups = np.array(groups)
    print(f"Final dataset shape: {X.shape}")

    out_path = os.path.join(PROCESSED_DIR, f"spectrograms_{channel}.npz")
    np.savez_compressed(out_path, X=X, y=y, groups=groups)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channel",
        choices=["mic", "vibration"],
        default="mic",
        help="Which sensor channel to build the dataset from (default: mic)",
    )
    args = parser.parse_args()
    main(channel=args.channel)
