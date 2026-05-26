# 저장된 SL/SSL 체크포인트를 불러와 여러 SNR 테스트셋에서 정확도를 평가하는 코드.

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
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


class SSLResNet(nn.Module):
    def __init__(self):
        super(SSLResNet, self).__init__()
        self.layer1 = ResBlock(2, 64)
        self.layer2 = ResBlock(64, 128)
        self.layer3 = ResBlock(128, 256)
        self.layer4 = ResBlock(256, 512)
        self.pool = nn.MaxPool1d(2)
        self.projection_head = nn.Sequential(
            nn.Linear(512 * 256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

    def forward(self, x):
        std = torch.std(x, dim=-1, keepdim=True) + 1e-8
        x = x / std
        x = self.pool(self.layer1(x))
        x = self.pool(self.layer2(x))
        x = self.pool(self.layer3(x))
        x = self.pool(self.layer4(x))
        x = x.view(x.size(0), -1)
        return self.projection_head(x)


class SLClassifier(nn.Module):
    def __init__(self, backbone, num_classes=5, seq_length=4096):
        super(SLClassifier, self).__init__()
        self.backbone = backbone
        feature_length = seq_length // 16
        self.fc = nn.Linear(512 * feature_length, num_classes)

    def forward(self, x):
        return self.fc(self.backbone.forward_features(x))


class SSLClassifier(nn.Module):
    def __init__(self, encoder, num_classes=5):
        super(SSLClassifier, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * 256, num_classes)

    def forward(self, x):
        std = torch.std(x, dim=-1, keepdim=True) + 1e-8
        x = x / std
        x = self.encoder.pool(self.encoder.layer1(x))
        x = self.encoder.pool(self.encoder.layer2(x))
        x = self.encoder.pool(self.encoder.layer3(x))
        x = self.encoder.pool(self.encoder.layer4(x))
        features = x.view(x.size(0), -1)
        return self.fc(self.dropout(features))


def parse_csv_list(value, item_type=str):
    return [item_type(item.strip()) for item in value.split(",") if item.strip()]


def evaluate_predictions(model, loader, device):
    preds = []
    labels = []
    model.eval()
    with torch.no_grad():
        for inputs, batch_labels in loader:
            outputs = model(inputs.to(device))
            preds.append(outputs.argmax(dim=1).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def build_metrics(args, modulations, confusion_rows):
    matrices = {}
    for row in confusion_rows:
        snr_key = f"{float(row['snr']):.1f}"
        true_label = int(row["true_label"])
        pred_label = int(row["pred_label"])
        count = int(row["count"])
        if snr_key not in matrices:
            matrices[snr_key] = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
        matrices[snr_key][true_label, pred_label] += count
    snr_acc = {}
    for key, matrix in matrices.items():
        total = int(matrix.sum())
        snr_acc[key] = float(np.trace(matrix) / total) if total else 0.0
    return {
        "config": {
            "experiment": args.experiment_name,
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "label_ratio": args.label_ratio,
            "num_per_class": args.num_per_class,
            "test_snrs": args.test_snrs,
        },
        "snr_acc": snr_acc,
        "snr_confusion_matrix": {key: value.tolist() for key, value in matrices.items()},
        "modulations": modulations,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Eval-only SNR sweep for saved SL/SSL checkpoints.")
    parser.add_argument("--model-type", choices=["sl", "ssl"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dir", default="test_data_snr")
    parser.add_argument("--test-snrs", default="-10,-5,0,1,2,3,4,5,10,15,20")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--modulations", default="BPSK,QPSK,8PSK,16QAM,64QAM")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--label-ratio", type=float, default=None)
    parser.add_argument("--num-per-class", type=int, default=200)
    parser.add_argument("--results-csv", default="eval_results.csv")
    parser.add_argument("--confusion-csv", default="eval_confusion.csv")
    parser.add_argument("--summary-json", default="eval_summary.json")
    parser.add_argument("--metrics-json", default="eval_metrics.json")
    parser.add_argument("--experiment-name", default="EVAL")
    return parser.parse_args()


args = parse_args()
modulations = parse_csv_list(args.modulations, str)
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.model_type == "sl":
    model = SLClassifier(SSLBackbone(), num_classes=args.num_classes, seq_length=args.seq_length).to(device)
else:
    model = SSLClassifier(SSLResNet(), num_classes=args.num_classes).to(device)

model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
model.eval()

test_rows = []
confusion_rows = []
for snr in parse_csv_list(args.test_snrs, int):
    x_path = os.path.join(args.test_dir, f"X_test_snr_{args.seq_length}_{snr}.npy")
    y_path = os.path.join(args.test_dir, f"Y_test_snr_{args.seq_length}_{snr}.npy")
    if not os.path.exists(x_path):
        print(f"Skipping missing test SNR {snr}: {x_path}")
        continue

    x_test = np.load(x_path).astype(np.float32)
    y_test = np.load(y_path).astype(np.int64)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    pred, labels = evaluate_predictions(model, loader, device)
    correct = int((pred == labels).sum())
    total = int(len(labels))
    acc = 100.0 * correct / total if total else 0.0
    print(f"SNR {snr:3d} dB | Accuracy: {acc:6.2f}%")

    test_rows.append(
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
            "checkpoint": args.checkpoint,
        }
    )
    for label in range(args.num_classes):
        mask = labels == label
        if not np.any(mask):
            continue
        mod_correct = int((pred[mask] == labels[mask]).sum())
        mod_total = int(mask.sum())
        test_rows.append(
            {
                "experiment": args.experiment_name,
                "snr": snr,
                "modulation": modulations[label] if label < len(modulations) else f"class_{label}",
                "label": label,
                "accuracy": 100.0 * mod_correct / mod_total if mod_total else 0.0,
                "correct": mod_correct,
                "total": mod_total,
                "label_ratio": args.label_ratio,
                "num_per_class": args.num_per_class,
                "checkpoint": args.checkpoint,
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
                    "true_modulation": modulations[true_label] if true_label < len(modulations) else f"class_{true_label}",
                    "pred_label": pred_label,
                    "pred_modulation": modulations[pred_label] if pred_label < len(modulations) else f"class_{pred_label}",
                    "count": count,
                    "true_total": true_total,
                    "row_percent": 100.0 * count / true_total if true_total else 0.0,
                    "label_ratio": args.label_ratio,
                    "num_per_class": args.num_per_class,
                    "checkpoint": args.checkpoint,
                }
            )

write_csv(
    args.results_csv,
    ["experiment", "snr", "modulation", "label", "accuracy", "correct", "total", "label_ratio", "num_per_class", "checkpoint"],
    test_rows,
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
        "checkpoint",
    ],
    confusion_rows,
)
write_json(
    args.summary_json,
    {
        "experiment": args.experiment_name,
        "model_type": args.model_type,
        "num_classes": args.num_classes,
        "seq_length": args.seq_length,
        "modulations": modulations,
        "label_ratio": args.label_ratio,
        "num_per_class": args.num_per_class,
        "checkpoint": args.checkpoint,
        "results_csv": args.results_csv,
        "confusion_csv": args.confusion_csv,
        "metrics_json": args.metrics_json,
    },
)
write_json(args.metrics_json, build_metrics(args, modulations, confusion_rows))

print(f"Saved results: {args.results_csv}")
print(f"Saved confusion: {args.confusion_csv}")
print(f"Saved summary: {args.summary_json}")
print(f"Saved metrics: {args.metrics_json}")
