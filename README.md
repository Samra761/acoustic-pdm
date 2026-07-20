# Acoustic Predictive Maintenance of Industrial Motors and Bearings

Spectrogram-based anomaly detection and fault classification from microphone audio, using the MAFAULDA (Machinery Fault Database) dataset. Built for the ITSOLERA AI internship, Project 02.

## Problem

Vibration-sensor based predictive maintenance is effective but expensive to deploy per-asset, which puts it out of reach for most small and medium manufacturers. This project tests whether a single microphone, combined with a deep learning pipeline, can flag developing motor and bearing faults using sound alone, no vibration hardware required.

## Dataset

[MAFAULDA](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html), a real recording set from a SpectraQuest Machinery Fault Simulator. Each 5-second sequence includes 8 sensor channels; this project uses only the microphone channel (the 8th column of each CSV). Six operating states are covered: normal, imbalance, horizontal misalignment, vertical misalignment, underhang bearing fault, and overhang bearing fault.

## Approach

1. **Feature extraction** — each recording is chunked into 1-second overlapping windows and converted to log-scaled Mel-spectrograms.
2. **Anomaly detection** — a convolutional autoencoder is trained only on spectrograms from the "normal" class. At inference time, reconstruction error on unseen audio becomes a failure-risk score: high error means the sound deviates from the machine's healthy baseline.
3. **Fault classification** — a separate CNN is trained on all six labeled classes, to identify which specific fault (bearing wear, imbalance, misalignment) is present once an anomaly is flagged.
4. **Evaluation** — ROC/AUROC for the anomaly detector (normal vs. all faults), and accuracy/F1/confusion matrix for the fault classifier. All train/test splits are done at the file level (not chunk level) so that overlapping windows from the same recording never leak between train and test.

## Repository structure

```
acoustic-pdm/
├── data/
│   ├── raw/            # unzipped MAFAULDA data goes here (not committed)
│   └── processed/      # generated spectrogram arrays (.npz)
├── src/
│   ├── data_loader.py       # scans raw CSVs, extracts microphone channel
│   ├── feature_extraction.py # audio -> Mel-spectrogram / MFCC
│   ├── build_dataset.py     # end-to-end feature extraction, saves .npz
│   ├── models.py            # ConvAutoencoder + FaultClassifier definitions
│   └── train.py             # trains both models, generates all evaluation plots
├── results/             # ROC curve, confusion matrix, classification report
├── models/              # saved model weights
└── requirements.txt
```

## Data

This repo does not include the raw MAFAULDA CSVs (too large for a code repo). The subset used here (90 files spanning all 6 classes) is hosted separately at [github.com/Samra761/mafaulda-subset-data](https://github.com/Samra761/mafaulda-subset-data). Clone that into `data/raw/` before running the pipeline, matching the folder structure described in `data_loader.py`.

## Running it

```bash
pip install -r requirements.txt

# 1. Clone the data subset into data/raw/ (see Data section above)
# 2. Build the spectrogram dataset
python src/build_dataset.py

# 3. Train and evaluate both models (cross-validated)
python src/train.py
```

Outputs land in `results/`: `autoencoder_roc.png`, `confusion_matrix.png`, `classification_report.txt`.

## Results

Evaluated with 3-fold group cross-validation (file-level folds, so overlapping chunks from one recording never appear in both train and test) given the modest number of source files in this subset.

**Anomaly detector (autoencoder, normal vs. all faults):** mean AUROC 0.938 ± 0.029 across folds. Reconstruction error cleanly separates healthy from faulty operation.

**Fault classifier (6-way):** mean accuracy 73.0% ± 7.7%, mean macro F1 0.57 ± 0.11 across folds. Overhang bearing fault, underhang bearing fault, and vertical misalignment are all classified reliably (F1 0.78–0.97). Horizontal misalignment and normal are the weaker classes, which traces directly to having only 8 source files each in this subset compared to 24 for the strongest classes — the model has proportionally less to learn from. See `results/confusion_matrix.png` for the full breakdown.

`results/autoencoder_roc.png` and `results/confusion_matrix.png` contain the plots; `results/classification_report.txt` has the full per-class precision/recall/F1.

## Scope and limitations

This project validates the acoustic modeling approach against a public benchmark dataset. It does not include the physical sensor deployment or the 2-4 week advance-warning field validation described as long-term objectives in the original project proposal, since those require live deployment on a running machine over time rather than a static, pre-recorded dataset. The dataset's severity levels (varying degrees of imbalance and misalignment) are used here as a proxy for fault progression rather than a true longitudinal time-to-failure signal.

## Reference

Ma, L., Zhang, Y., & Wang, Z. (2025). Fault Diagnosis of Motor Bearing Transmission System Based on Acoustic Characteristics. *Sensors*, 26(1), 259.
