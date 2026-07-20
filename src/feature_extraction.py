"""
feature_extraction.py
Converts raw microphone signals into Mel-spectrogram and MFCC representations,
per Objective 2 of the proposal ("transform raw audio into spectrogram, scalogram,
and MFCC representations").

Each 5-second MAFAULDA sequence is chunked into fixed-length windows to
(a) generate more training examples and (b) mimic real-time streaming inference.
"""

import numpy as np
import librosa

WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.2
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
TARGET_TIME_FRAMES = 196  # fixed width so all spectrograms are the same shape
DB_FLOOR = -50.0  # fixed global dB range for normalization (calibrated against this dataset's actual range of ~-42 to +10 dB)
DB_CEIL = 15.0


def chunk_signal(signal, sample_rate, window_sec=WINDOW_SECONDS, hop_sec=HOP_SECONDS):
    """Splits a long signal into overlapping fixed-length windows."""
    window_len = int(window_sec * sample_rate)
    hop_len = int(hop_sec * sample_rate)
    chunks = []
    for start in range(0, len(signal) - window_len + 1, hop_len):
        chunks.append(signal[start:start + window_len])
    return chunks


def signal_to_mel_spectrogram(signal, sample_rate):
    """Converts a 1D audio chunk into a log-scaled Mel-spectrogram, fixed shape.

    IMPORTANT: uses a fixed reference power (ref=1.0) rather than librosa's
    default ref=np.max, and a fixed global dB range for normalization rather
    than per-sample min/max. Per-sample normalization would rescale every
    spectrogram to occupy the same [0,1] range regardless of its absolute
    loudness, which throws away exactly the signal the anomaly detector needs
    (faults are often louder/quieter in absolute terms, not just differently
    shaped). Fixed global scaling keeps that information intact.
    """
    mel = librosa.feature.melspectrogram(
        y=signal, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=1.0)

    # pad or truncate to fixed time frames
    if log_mel.shape[1] < TARGET_TIME_FRAMES:
        pad_width = TARGET_TIME_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=DB_FLOOR)
    else:
        log_mel = log_mel[:, :TARGET_TIME_FRAMES]

    # fixed global normalization (not per-sample) so absolute energy is preserved
    log_mel = np.clip(log_mel, DB_FLOOR, DB_CEIL)
    log_mel = (log_mel - DB_FLOOR) / (DB_CEIL - DB_FLOOR)
    return log_mel.astype(np.float32)


def signal_to_mfcc(signal, sample_rate, n_mfcc=20):
    """Converts a 1D audio chunk into MFCCs (used as an auxiliary/classical baseline feature)."""
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return mfcc.astype(np.float32)
