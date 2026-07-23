"""
data_loader.py
Loads MAFAULDA CSV files, extracts a chosen sensor channel (microphone or
vibration), and builds a labeled index of all sequences.

Expected raw data layout after unzipping the Kaggle/UFRJ archive:
data/raw/
    normal/*.csv
    imbalance/*.csv
    horizontal-misalignment/*.csv
    vertical-misalignment/*.csv
    underhang/{ball,cage,outer_race}/*.csv
    overhang/{ball,cage,outer_race}/*.csv

If your extracted folder names differ, edit CLASS_FOLDER_MAP below to match.
"""

import os
import glob
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Map: label name -> list of relative folder patterns to search under RAW_DIR
CLASS_FOLDER_MAP = {
    "normal": ["normal"],
    "imbalance": ["imbalance"],
    "horizontal_misalignment": ["horizontal-misalignment", "horizontal_misalignment"],
    "vertical_misalignment": ["vertical-misalignment", "vertical_misalignment"],
    "underhang_bearing_fault": ["underhang/*", "underhang_*"],
    "overhang_bearing_fault": ["overhang/*", "overhang_*"],
}

MIC_COLUMN_INDEX = 7        # 8th column (0-indexed) is the microphone signal
VIBRATION_COLUMN_INDEX = 2  # underhang bearing, radial direction


def build_file_index():
    """Scans RAW_DIR and returns a DataFrame of [filepath, label]."""
    rows = []
    for label, patterns in CLASS_FOLDER_MAP.items():
        for pattern in patterns:
            search_path = os.path.join(RAW_DIR, pattern, "**", "*.csv")
            files = glob.glob(search_path, recursive=True)
            for f in files:
                rows.append({"filepath": f, "label": label})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"No CSV files found under {RAW_DIR}. "
            "Confirm the dataset has been unzipped there and folder names "
            "match CLASS_FOLDER_MAP (edit the map if the archive uses different names)."
        )
    return df


def load_audio_signal(filepath, sample_rate=50000, channel="mic"):
    """Reads a single MAFAULDA CSV and returns the requested sensor channel as a 1D array.

    channel: 'mic' for microphone (default) or 'vibration' for underhang radial accelerometer.
    """
    column_map = {"mic": MIC_COLUMN_INDEX, "vibration": VIBRATION_COLUMN_INDEX}
    if channel not in column_map:
        raise ValueError(f"channel must be one of {list(column_map)}, got '{channel}'")
    df = pd.read_csv(filepath, header=None)
    signal = df.iloc[:, column_map[channel]].to_numpy(dtype=np.float32)
    return signal, sample_rate


if __name__ == "__main__":
    index = build_file_index()
    print(index["label"].value_counts())
    print(f"Total sequences found: {len(index)}")
    index.to_csv(os.path.join(RAW_DIR, "..", "file_index.csv"), index=False)