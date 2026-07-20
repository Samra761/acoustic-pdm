"""
train.py
Trains both models and produces the evaluation artifacts for the report/README.

Given the small file count in this MAFAULDA subset (as few as 8 source files
for some classes), a single train/test split is unstable: which few files
happen to land in the test set can swing accuracy by 20+ points between runs.
Instead, this script uses GROUP K-FOLD CROSS-VALIDATION (file-level groups,
so overlapping chunks from one recording never span train and test) and
reports the mean +/- std across folds, plus a confusion matrix pooled over
all folds. This is standard practice for small datasets and gives a much more
defensible, reproducible result than a lucky/unlucky single split.

- Convolutional autoencoder on "normal"-only spectrograms -> anomaly score
  (ROC curve + AUROC on normal vs all-faults, Objective 2 & 4)
- CNN fault classifier on all 6 classes -> confusion matrix, accuracy, F1
  (Objective 3)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from models import ConvAutoencoder, FaultClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

N_FOLDS = 3  # kept low because the smallest class has only 8 source files
EPOCHS = 6
BATCH_SIZE = 64
LR = 1e-3


class SpecDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)
        if self.y is None:
            return x
        return x, self.y[idx]


def load_data():
    data = np.load(os.path.join(PROCESSED_DIR, "spectrograms.npz"), allow_pickle=True)
    return data["X"], data["y"], data["groups"]


def make_loader(X, y=None, batch_size=BATCH_SIZE, shuffle=False):
    ds = SpecDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Stage 1: Autoencoder, cross-validated
# ---------------------------------------------------------------------------
def train_autoencoder_once(X_train_normal):
    model = ConvAutoencoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    loader = make_loader(X_train_normal, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            opt.step()
    return model


def score_autoencoder(model, X):
    model.eval()
    loader = make_loader(X)
    errors = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            err = ((recon - batch) ** 2).mean(dim=(1, 2, 3))
            errors.extend(err.cpu().numpy())
    return np.array(errors)


def cross_validate_autoencoder(X, y, groups):
    """
    K-fold over the NORMAL files only (that's all this model trains on).
    In each fold, the held-out normal files + every fault-class spectrogram
    are scored together to measure normal-vs-fault separability.
    """
    is_normal = y == "normal"
    X_normal, groups_normal = X[is_normal], groups[is_normal]
    X_fault = X[~is_normal]

    n_normal_groups = len(set(groups_normal))
    k = min(N_FOLDS, n_normal_groups)
    gkf = GroupKFold(n_splits=k)

    fold_aurocs = []
    all_true_labels, all_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_normal, groups=groups_normal)):
        model = train_autoencoder_once(X_normal[train_idx])
        held_out_normal = X_normal[val_idx]

        eval_X = np.concatenate([held_out_normal, X_fault])
        eval_labels = np.concatenate([
            np.zeros(len(held_out_normal)),
            np.ones(len(X_fault)),
        ])
        scores = score_autoencoder(model, eval_X)

        fold_auroc = roc_auc_score(eval_labels, scores)
        fold_aurocs.append(fold_auroc)
        all_true_labels.append(eval_labels)
        all_scores.append(scores)
        print(f"[Autoencoder] Fold {fold+1}/{k} - held-out normal files: {len(val_idx)} - AUROC: {fold_auroc:.4f}")

    mean_auroc, std_auroc = np.mean(fold_aurocs), np.std(fold_aurocs)
    print(f"[Autoencoder] Mean AUROC across {k} folds: {mean_auroc:.4f} +/- {std_auroc:.4f}")

    pooled_labels = np.concatenate(all_true_labels)
    pooled_scores = np.concatenate(all_scores)
    fpr, tpr, _ = roc_curve(pooled_labels, pooled_scores)
    pooled_auroc = roc_auc_score(pooled_labels, pooled_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Pooled AUROC = {pooled_auroc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Anomaly Detector ROC (pooled over {k}-fold CV)\nMean fold AUROC: {mean_auroc:.4f} +/- {std_auroc:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "autoencoder_roc.png"), dpi=150)
    plt.close()

    final_model = train_autoencoder_once(X_normal)
    torch.save(final_model.state_dict(), os.path.join(MODELS_DIR, "autoencoder.pt"))

    return mean_auroc, std_auroc


# ---------------------------------------------------------------------------
# Stage 2: Fault classifier, cross-validated
# ---------------------------------------------------------------------------
def train_classifier_once(X_train, y_train, num_classes):
    model = FaultClassifier(num_classes=num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(X_train, y_train, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).long()
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
    return model


def predict_classifier(model, X):
    model.eval()
    loader = make_loader(X)
    preds = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            preds.extend(logits.argmax(1).cpu().numpy())
    return np.array(preds)


def cross_validate_classifier(X, y, groups):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    k = min(N_FOLDS, len(set(groups)))
    gkf = GroupKFold(n_splits=k)

    fold_accs, fold_f1s = [], []
    all_true, all_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_enc, groups=groups)):
        model = train_classifier_once(X[train_idx], y_enc[train_idx], num_classes)
        preds = predict_classifier(model, X[test_idx])
        true = y_enc[test_idx]

        acc = accuracy_score(true, preds)
        f1 = f1_score(true, preds, average="macro", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        all_true.extend(true)
        all_pred.extend(preds)
        print(f"[Classifier] Fold {fold+1}/{k} - accuracy: {acc:.4f} - macro F1: {f1:.4f}")

    mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
    mean_f1, std_f1 = np.mean(fold_f1s), np.std(fold_f1s)
    print(f"[Classifier] Mean accuracy across {k} folds: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"[Classifier] Mean macro F1 across {k} folds: {mean_f1:.4f} +/- {std_f1:.4f}")

    report = classification_report(all_true, all_pred, target_names=le.classes_, digits=4, zero_division=0)
    print(report)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(f"{k}-fold cross-validated results (pooled predictions across all folds)\n")
        f.write(f"Mean accuracy: {mean_acc:.4f} +/- {std_acc:.4f}\n")
        f.write(f"Mean macro F1: {mean_f1:.4f} +/- {std_f1:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Fault Classifier Confusion Matrix (pooled, {k}-fold CV)\nMean accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    final_model = train_classifier_once(X, y_enc, num_classes)
    torch.save(final_model.state_dict(), os.path.join(MODELS_DIR, "classifier.pt"))

    return mean_acc, std_acc, mean_f1, std_f1, le


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    X, y, groups = load_data()
    print(f"Loaded {X.shape[0]} spectrograms across {len(set(y))} classes, {len(set(groups))} source files.")

    print("\n=== Cross-validating anomaly detector (autoencoder) ===")
    cross_validate_autoencoder(X, y, groups)

    print("\n=== Cross-validating fault classifier ===")
    cross_validate_classifier(X, y, groups)

    print("\nDone. See results/ for ROC curve, confusion matrix, and classification report.")
