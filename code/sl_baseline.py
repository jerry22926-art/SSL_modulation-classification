# 라벨이 있는 일부 데이터만 사용해 supervised learning baseline을 학습하고 평가하는 코드.

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class SSLBackbone(nn.Module):
    def __init__(self):
        super(SSLBackbone, self).__init__()
        self.layer1 = ResBlock(2, 64)
        self.layer2 = ResBlock(64, 128)
        self.layer3 = ResBlock(128, 256)
        self.layer4 = ResBlock(256, 512)
        self.pool = nn.MaxPool1d(2)

    def forward_features(self, x):
        std = torch.std(x, dim=-1, keepdim=True) + 1e-8
        x = x / std
        x = self.pool(self.layer1(x))
        x = self.pool(self.layer2(x))
        x = self.pool(self.layer3(x))
        x = self.pool(self.layer4(x))
        return x.view(x.size(0), -1)

    def forward(self, x):
        return self.forward_features(x)


class SLClassifier(nn.Module):
    def __init__(self, backbone, num_classes=5, seq_length=4096):
        super(SLClassifier, self).__init__()
        self.backbone = backbone
        feature_length = seq_length // 16
        self.fc = nn.Linear(512 * feature_length, num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        return self.fc(features)


def parse_csv_list(value, item_type=str):
    return [item_type(item.strip()) for item in value.split(",") if item.strip()]


def stratified_sample(x_data, y_label, num_classes, num_per_class=None, label_ratio=None, snr_label=None):
    indices = np.arange(len(x_data))
    np.random.shuffle(indices)

    x_list, y_list = [], []
    if snr_label is None:
        group_keys = [(c, None) for c in range(num_classes)]
    else:
        group_keys = [(c, snr) for c in range(num_classes) for snr in np.unique(snr_label)]

    for c, snr in group_keys:
        if snr is None:
            group_idx = indices[y_label[indices] == c]
        else:
            group_idx = indices[(y_label[indices] == c) & (snr_label[indices] == snr)]
        if len(group_idx) == 0:
            continue

        if label_ratio is not None:
            sample_count = max(1, int(len(group_idx) * label_ratio))
        else:
            sample_count = min(num_per_class, len(group_idx))

        selected_idx = group_idx[:sample_count]
        x_list.append(x_data[selected_idx])
        y_list.append(y_label[selected_idx])

    return np.concatenate(x_list), np.concatenate(y_list)


def load_train_val_data(args):
    split_paths = [args.train_x, args.train_y, args.val_x, args.val_y]
    if all(os.path.exists(path) for path in split_paths):
        snr_train = np.load(args.train_snr).astype(np.int64) if os.path.exists(args.train_snr) else None
        snr_val = np.load(args.val_snr).astype(np.int64) if os.path.exists(args.val_snr) else None
        return (
            np.load(args.train_x).astype(np.float32),
            np.load(args.train_y).astype(np.int64),
            snr_train,
            np.load(args.val_x).astype(np.float32),
            np.load(args.val_y).astype(np.int64),
            snr_val,
        )

    x_data = np.load(args.x_data).astype(np.float32)
    y_label = np.load(args.y_label).astype(np.int64)
    snr_label = np.load(args.snr_label).astype(np.int64) if os.path.exists(args.snr_label) else None
    train_idx, val_idx = [], []

    if snr_label is None:
        group_keys = [(c, None) for c in np.unique(y_label)]
    else:
        group_keys = [(c, snr) for c in np.unique(y_label) for snr in np.unique(snr_label)]

    for c, snr in group_keys:
        if snr is None:
            group_idx = np.where(y_label == c)[0]
        else:
            group_idx = np.where((y_label == c) & (snr_label == snr))[0]
        if len(group_idx) == 0:
            continue
        np.random.shuffle(group_idx)
        val_count = max(1, int(len(group_idx) * 0.2))
        val_idx.extend(group_idx[:val_count])
        train_idx.extend(group_idx[val_count:])

    return (
        x_data[train_idx],
        y_label[train_idx],
        None if snr_label is None else snr_label[train_idx],
        x_data[val_idx],
        y_label[val_idx],
        None if snr_label is None else snr_label[val_idx],
    )


def evaluate_predictions(model, data_loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for inputs, batch_labels in data_loader:
            outputs = model(inputs.to(device))
            _, batch_preds = torch.max(outputs, 1)
            preds.append(batch_preds.cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def evaluate_accuracy(model, data_loader, device):
    pred, labels = evaluate_predictions(model, data_loader, device)
    return 100 * float((pred == labels).sum()) / len(labels)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def build_metrics_payload(args, modulations, history, confusion_rows, best_val_acc, best_path, last_path):
    matrix = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    snr_matrices = {}

    for row in confusion_rows:
        snr_key = f"{float(row['snr']):.1f}"
        true_label = int(row["true_label"])
        pred_label = int(row["pred_label"])
        count = int(row["count"])
        matrix[true_label, pred_label] += count
        if snr_key not in snr_matrices:
            snr_matrices[snr_key] = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
        snr_matrices[snr_key][true_label, pred_label] += count

    total = int(matrix.sum())
    test_acc = float(np.trace(matrix) / total) if total else 0.0
    snr_acc = {
        key: float(np.trace(value) / int(value.sum())) if int(value.sum()) else 0.0
        for key, value in snr_matrices.items()
    }
    mod_acc = {}
    for label in range(args.num_classes):
        label_total = int(matrix[label].sum())
        name = modulations[label] if label < len(modulations) else f"class_{label}"
        mod_acc[name] = float(matrix[label, label] / label_total) if label_total else 0.0

    return {
        "best_val_acc": best_val_acc / 100,
        "test_acc": test_acc,
        "snr_acc": snr_acc,
        "mod_acc": mod_acc,
        "confusion_matrix": matrix.tolist(),
        "snr_confusion_matrix": {key: value.tolist() for key, value in snr_matrices.items()},
        "history": [
            {
                "epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "val_acc": row["val_accuracy"] / 100,
            }
            for row in history
        ],
        "config": {
            "experiment": args.experiment_name,
            "model": "SL supervised baseline with SSL backbone",
            "x_data": args.x_data,
            "y_label": args.y_label,
            "snr_label": args.snr_label,
            "train_x": args.train_x,
            "train_y": args.train_y,
            "train_snr": args.train_snr,
            "val_x": args.val_x,
            "val_y": args.val_y,
            "val_snr": args.val_snr,
            "test_dir": args.test_dir,
            "test_snrs": args.test_snrs,
            "modulations": modulations,
            "num_classes": args.num_classes,
            "seq_length": args.seq_length,
            "label_ratio": args.label_ratio,
            "num_per_class": args.num_per_class,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "encoder_lr": args.encoder_lr,
            "classifier_lr": args.classifier_lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "best_checkpoint": best_path,
            "last_checkpoint": last_path,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised baseline using the same backbone as SSL.")
    parser.add_argument("--x-data", default="X_train_4096.npy")
    parser.add_argument("--y-label", default="Y_train_4096.npy")
    parser.add_argument("--snr-label", default="SNR_train_4096.npy")
    parser.add_argument("--train-x", default="X_train_4096.npy")
    parser.add_argument("--train-y", default="Y_train_4096.npy")
    parser.add_argument("--train-snr", default="SNR_train_4096.npy")
    parser.add_argument("--val-x", default="X_val_4096.npy")
    parser.add_argument("--val-y", default="Y_val_4096.npy")
    parser.add_argument("--val-snr", default="SNR_val_4096.npy")
    parser.add_argument("--test-dir", default="test_data_snr")
    parser.add_argument("--test-snrs", default="-10,-5,0,5,10,15,20")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--modulations", default="BPSK,QPSK,8PSK,16QAM,64QAM")
    parser.add_argument("--num-per-class", type=int, default=200)
    parser.add_argument("--label-ratio", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--classifier-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--best-path", default="best_sl_baseline.pth")
    parser.add_argument("--last-path", default="last_sl_baseline.pth")
    parser.add_argument("--results-csv", default="sl_results.csv")
    parser.add_argument("--confusion-csv", default="sl_confusion.csv")
    parser.add_argument("--log-csv", default="sl_train_log.csv")
    parser.add_argument("--summary-json", default="sl_summary.json")
    parser.add_argument("--metrics-json", default="sl_metrics.json")
    parser.add_argument("--experiment-name", default="SL")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


args = parse_args()
modulations = parse_csv_list(args.modulations, str)
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

X_train_raw, Y_train_raw, SNR_train_raw, X_val_raw, Y_val_raw, _ = load_train_val_data(args)
X_train_small, Y_train_small = stratified_sample(
    X_train_raw,
    Y_train_raw,
    args.num_classes,
    num_per_class=args.num_per_class,
    label_ratio=args.label_ratio,
    snr_label=SNR_train_raw,
)

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train_small), torch.from_numpy(Y_train_small)),
    batch_size=args.batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val_raw), torch.from_numpy(Y_val_raw)),
    batch_size=args.batch_size,
    shuffle=False,
)

model = SLClassifier(SSLBackbone(), num_classes=args.num_classes, seq_length=args.seq_length).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    [
        {"params": model.backbone.parameters(), "lr": args.encoder_lr},
        {"params": model.fc.parameters(), "lr": args.classifier_lr},
    ],
    weight_decay=args.weight_decay,
)

print(
    f"SL baseline training start: {len(X_train_small)} labeled samples, "
    f"{args.num_classes} classes, label_ratio={args.label_ratio}, "
    f"batch_size={args.batch_size}, encoder_lr={args.encoder_lr}, "
    f"classifier_lr={args.classifier_lr}, weight_decay={args.weight_decay}"
)

best_val_acc = 0.0
history = []
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    val_acc = evaluate_accuracy(model, val_loader, device)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), args.best_path)

    history.append(
        {
            "experiment": args.experiment_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "best_val_accuracy": best_val_acc,
        }
    )
    print(
        f"Epoch {epoch + 1}/{args.epochs} - Loss: {train_loss:.4f}, "
        f"Val Acc: {val_acc:.2f}%, Best: {best_val_acc:.2f}%"
    )

torch.save(model.state_dict(), args.last_path)

write_csv(
    args.log_csv,
    ["experiment", "epoch", "train_loss", "val_accuracy", "best_val_accuracy"],
    history,
)

print("\n" + "=" * 50)
print(f"SL baseline final result using best val checkpoint ({best_val_acc:.2f}%)")
print("=" * 50)

model.load_state_dict(torch.load(args.best_path, map_location=device, weights_only=True))
model.eval()

result_rows = []
confusion_rows = []
for snr in parse_csv_list(args.test_snrs, int):
    x_path = os.path.join(args.test_dir, f"X_test_snr_{args.seq_length}_{snr}.npy")
    y_path = os.path.join(args.test_dir, f"Y_test_snr_{args.seq_length}_{snr}.npy")
    if not os.path.exists(x_path):
        continue

    X_test = np.load(x_path).astype(np.float32)
    Y_test = np.load(y_path).astype(np.int64)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    pred, labels = evaluate_predictions(model, test_loader, device)
    correct = int((pred == labels).sum())
    total = int(len(labels))
    acc = 100 * correct / total
    print(f"SNR {snr:3d} dB | Accuracy: {acc:6.2f}%")

    result_rows.append(
        {
            "experiment": args.experiment_name,
            "snr": snr,
            "modulation": "ALL",
            "label": "ALL",
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "label_ratio": args.label_ratio,
            "num_per_class": args.num_per_class,
            "epochs": args.epochs,
            "checkpoint": args.best_path,
        }
    )

    for label in range(args.num_classes):
        mask = labels == label
        if not np.any(mask):
            continue
        mod_correct = int((pred[mask] == labels[mask]).sum())
        mod_total = int(mask.sum())
        result_rows.append(
            {
                "experiment": args.experiment_name,
                "snr": snr,
                "modulation": modulations[label] if label < len(modulations) else f"class_{label}",
                "label": label,
                "accuracy": 100 * mod_correct / mod_total,
                "correct": mod_correct,
                "total": mod_total,
                "label_ratio": args.label_ratio,
                "num_per_class": args.num_per_class,
                "epochs": args.epochs,
                "checkpoint": args.best_path,
            }
        )

    for true_label in range(args.num_classes):
        true_mask = labels == true_label
        true_total = int(true_mask.sum())
        if true_total == 0:
            continue
        for pred_label in range(args.num_classes):
            count = int(((labels == true_label) & (pred == pred_label)).sum())
            confusion_rows.append(
                {
                    "experiment": args.experiment_name,
                    "snr": snr,
                    "true_label": true_label,
                    "true_modulation": modulations[true_label]
                    if true_label < len(modulations)
                    else f"class_{true_label}",
                    "pred_label": pred_label,
                    "pred_modulation": modulations[pred_label]
                    if pred_label < len(modulations)
                    else f"class_{pred_label}",
                    "count": count,
                    "true_total": true_total,
                    "row_percent": 100 * count / true_total,
                    "label_ratio": args.label_ratio,
                    "num_per_class": args.num_per_class,
                    "epochs": args.epochs,
                    "checkpoint": args.best_path,
                }
            )

write_csv(
    args.results_csv,
    [
        "experiment",
        "snr",
        "modulation",
        "label",
        "accuracy",
        "correct",
        "total",
        "label_ratio",
        "num_per_class",
        "epochs",
        "checkpoint",
    ],
    result_rows,
)
write_csv(
    args.confusion_csv,
    [
        "experiment",
        "snr",
        "true_label",
        "true_modulation",
        "pred_label",
        "pred_modulation",
        "count",
        "true_total",
        "row_percent",
        "label_ratio",
        "num_per_class",
        "epochs",
        "checkpoint",
    ],
    confusion_rows,
)

summary = {
    "experiment": args.experiment_name,
    "model": "SL supervised baseline with SSL backbone",
    "num_classes": args.num_classes,
    "seq_length": args.seq_length,
    "modulations": modulations,
    "label_ratio": args.label_ratio,
    "num_per_class": args.num_per_class,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "encoder_lr": args.encoder_lr,
    "classifier_lr": args.classifier_lr,
    "weight_decay": args.weight_decay,
    "best_val_accuracy": best_val_acc,
    "best_checkpoint": args.best_path,
    "last_checkpoint": args.last_path,
    "results_csv": args.results_csv,
    "confusion_csv": args.confusion_csv,
    "log_csv": args.log_csv,
}
write_json(args.summary_json, summary)
write_json(
    args.metrics_json,
    build_metrics_payload(args, modulations, history, confusion_rows, best_val_acc, args.best_path, args.last_path),
)

print(f"Saved results: {args.results_csv}")
print(f"Saved confusion matrix: {args.confusion_csv}")
print(f"Saved train log: {args.log_csv}")
print(f"Saved summary: {args.summary_json}")
print(f"Saved metrics JSON: {args.metrics_json}")
