# 공유 SSL encoder 위에 SNR regime별 classifier head를 붙여 per-head SSL을 학습하고 평가하는 코드.

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
        super().__init__()
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


class SSLResNet(nn.Module):
    def __init__(self):
        super().__init__()
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

    def forward_features(self, x):
        std = torch.std(x, dim=-1, keepdim=True) + 1e-8
        x = x / std
        x = self.pool(self.layer1(x))
        x = self.pool(self.layer2(x))
        x = self.pool(self.layer3(x))
        x = self.pool(self.layer4(x))
        return x.view(x.size(0), -1)

    def forward(self, x):
        return self.projection_head(self.forward_features(x))


class MultiHeadClassifier(nn.Module):
    def __init__(self, encoder, num_classes=5, feature_dim=512 * 256, num_heads=3, dropout=0.5):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(feature_dim, num_classes) for _ in range(num_heads)])

    def forward(self, x, group_ids):
        features = self.dropout(self.encoder.forward_features(x))
        logits = torch.empty(
            (x.size(0), self.heads[0].out_features),
            dtype=features.dtype,
            device=features.device,
        )
        for group_idx, head in enumerate(self.heads):
            mask = group_ids == group_idx
            if mask.any():
                logits[mask] = head(features[mask])
        return logits


class SimpleAug:
    def __call__(self, x):
        x_aug = x.clone().float()
        if np.random.rand() < 0.5:
            x_aug = x_aug + torch.randn_like(x_aug) * 0.01
        return x_aug


def parse_csv_list(value, item_type=str):
    return [item_type(item.strip()) for item in value.split(",") if item.strip()]


def snr_to_group_numpy(snr_values):
    snr_values = np.asarray(snr_values)
    group = np.zeros_like(snr_values, dtype=np.int64)
    group[snr_values >= 0] = 1
    group[snr_values >= 10] = 2
    return group


def snr_to_group_scalar(snr):
    if snr >= 10:
        return 2
    if snr >= 0:
        return 1
    return 0


def group_name(group_id):
    return ["low", "transition", "high"][group_id]


def per_head_balanced_sample(x_data, y_label, snr_label, num_classes, samples_per_head):
    x_parts, y_parts, snr_parts, group_parts = [], [], [], []
    summary = []
    all_indices = np.arange(len(x_data))

    for group_id in range(3):
        group_snrs = np.unique(snr_label[snr_to_group_numpy(snr_label) == group_id])
        keys = [(c, snr) for c in range(num_classes) for snr in group_snrs]
        base = samples_per_head // len(keys)
        remainder = samples_per_head % len(keys)
        selected_group = []

        for key_idx, (label, snr) in enumerate(keys):
            target = base + (1 if key_idx < remainder else 0)
            candidates = all_indices[(y_label == label) & (snr_label == snr)]
            if len(candidates) == 0:
                continue
            np.random.shuffle(candidates)
            take = min(target, len(candidates))
            selected = candidates[:take]
            selected_group.append(selected)
            summary.append(
                {
                    "group": group_name(group_id),
                    "group_id": group_id,
                    "snr": int(snr),
                    "label": int(label),
                    "selected": int(take),
                    "available": int(len(candidates)),
                }
            )

        selected_group = np.concatenate(selected_group)
        np.random.shuffle(selected_group)
        x_parts.append(x_data[selected_group])
        y_parts.append(y_label[selected_group])
        snr_parts.append(snr_label[selected_group])
        group_parts.append(np.full(len(selected_group), group_id, dtype=np.int64))

    return (
        np.concatenate(x_parts),
        np.concatenate(y_parts),
        np.concatenate(snr_parts),
        np.concatenate(group_parts),
        summary,
    )


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, group in loader:
            pred = model(x.to(device), group.to(device)).argmax(dim=1).cpu()
            correct += int((pred == y).sum().item())
            total += y.size(0)
    return 100.0 * correct / total if total else 0.0


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def build_metrics_payload(args, modulations, train_log_rows, confusion_rows, best_val_acc, sample_summary):
    matrices = {}
    for row in confusion_rows:
        key = f"{float(row['snr']):.1f}"
        if key not in matrices:
            matrices[key] = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
        matrices[key][int(row["true_label"]), int(row["pred_label"])] += int(row["count"])

    all_matrix = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    for matrix in matrices.values():
        all_matrix += matrix

    return {
        "best_val_acc": best_val_acc / 100.0,
        "test_acc": float(np.trace(all_matrix) / all_matrix.sum()) if all_matrix.sum() else 0.0,
        "snr_acc": {
            key: float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0
            for key, matrix in matrices.items()
        },
        "confusion_matrix": all_matrix.tolist(),
        "snr_confusion_matrix": {key: value.tolist() for key, value in matrices.items()},
        "sample_summary": sample_summary,
        "history": [
            {"epoch": row["epoch"], "train_loss": row["train_loss"], "val_acc": row["val_accuracy"] / 100.0}
            for row in train_log_rows
        ],
        "config": {
            "experiment": args.experiment_name,
            "pretrained_path": args.pretrained_path,
            "samples_per_head": args.samples_per_head,
            "total_labeled": args.samples_per_head * 3,
            "snr_groups": {
                "low": "<0, trained from -10/-5 dB",
                "transition": "0..5, trained from 0/5 dB",
                "high": ">=10, trained from 10/15/20 dB in default train set",
            },
            "train_snrs": sorted({int(item["snr"]) for item in sample_summary}),
            "test_snrs": args.test_snrs,
            "modulations": modulations,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="SNR multi-head SSL fine-tuning with equal samples per head.")
    parser.add_argument("--pretrained-path", required=True)
    parser.add_argument("--train-x", default="X_train_4096.npy")
    parser.add_argument("--train-y", default="Y_train_4096.npy")
    parser.add_argument("--train-snr", default="SNR_train_4096.npy")
    parser.add_argument("--val-x", default="X_val_4096.npy")
    parser.add_argument("--val-y", default="Y_val_4096.npy")
    parser.add_argument("--val-snr", default="SNR_val_4096.npy")
    parser.add_argument("--test-dir", default="test_data_snr")
    parser.add_argument("--test-snrs", default="-10,-5,0,1,2,3,4,5,10,15,20")
    parser.add_argument("--modulations", default="BPSK,QPSK,8PSK,16QAM,64QAM")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--samples-per-head", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--classifier-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--best-path", default="best_per_head_multihead.pth")
    parser.add_argument("--last-path", default="last_per_head_multihead.pth")
    parser.add_argument("--results-csv", default="per_head_multihead_results.csv")
    parser.add_argument("--confusion-csv", default="per_head_multihead_confusion.csv")
    parser.add_argument("--log-csv", default="per_head_multihead_train_log.csv")
    parser.add_argument("--summary-json", default="per_head_multihead_summary.json")
    parser.add_argument("--metrics-json", default="per_head_multihead_metrics.json")
    parser.add_argument("--experiment-name", default="SSL_PER_HEAD_MULTIHEAD")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    modulations = parse_csv_list(args.modulations, str)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_train = np.load(args.train_x).astype(np.float32)
    y_train = np.load(args.train_y).astype(np.int64)
    snr_train = np.load(args.train_snr).astype(np.int64)
    x_val = np.load(args.val_x).astype(np.float32)
    y_val = np.load(args.val_y).astype(np.int64)
    snr_val = np.load(args.val_snr).astype(np.int64)

    x_small, y_small, snr_small, group_small, sample_summary = per_head_balanced_sample(
        x_train,
        y_train,
        snr_train,
        args.num_classes,
        args.samples_per_head,
    )
    val_groups = snr_to_group_numpy(snr_val)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_small), torch.from_numpy(y_small), torch.from_numpy(group_small)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val), torch.from_numpy(val_groups)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    encoder = SSLResNet().to(device)
    encoder.load_state_dict(torch.load(args.pretrained_path, map_location=device, weights_only=True))
    model = MultiHeadClassifier(encoder, num_classes=args.num_classes, num_heads=3).to(device)
    optimizer = optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.heads.parameters(), "lr": args.classifier_lr},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    aug = SimpleAug()

    print(f"Training {args.experiment_name}: {len(x_small)} labeled samples ({args.samples_per_head}/head)")
    best_val_acc = 0.0
    train_log_rows = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for x, y, group in train_loader:
            x_aug = torch.stack([aug(item) for item in x]).to(device)
            y = y.to(device)
            group = group.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_aug, group), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_acc = evaluate_accuracy(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.best_path)
        train_log_rows.append(
            {
                "experiment": args.experiment_name,
                "epoch": epoch + 1,
                "train_loss": epoch_loss / len(train_loader),
                "val_accuracy": val_acc,
                "best_val_accuracy": best_val_acc,
            }
        )
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:04d} | Loss {epoch_loss / len(train_loader):.4f} | Val {val_acc:.2f}% | Best {best_val_acc:.2f}%")
        if device == "cuda":
            torch.cuda.empty_cache()

    torch.save(model.state_dict(), args.last_path)
    write_csv(args.log_csv, ["experiment", "epoch", "train_loss", "val_accuracy", "best_val_accuracy"], train_log_rows)

    model.load_state_dict(torch.load(args.best_path, map_location=device, weights_only=True))
    model.eval()
    test_rows, confusion_rows = [], []
    for snr in parse_csv_list(args.test_snrs, int):
        x_path = os.path.join(args.test_dir, f"X_test_snr_4096_{snr}.npy")
        y_path = os.path.join(args.test_dir, f"Y_test_snr_4096_{snr}.npy")
        if not os.path.exists(x_path):
            print(f"Skipping missing SNR {snr}: {x_path}")
            continue
        x_test = torch.from_numpy(np.load(x_path).astype(np.float32))
        y_test = np.load(y_path).astype(np.int64)
        group_id = snr_to_group_scalar(snr)
        loader = DataLoader(TensorDataset(x_test), batch_size=args.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for (x_batch,) in loader:
                groups = torch.full((x_batch.size(0),), group_id, dtype=torch.long, device=device)
                preds.append(model(x_batch.to(device), groups).argmax(dim=1).cpu().numpy())
        pred = np.concatenate(preds)
        correct = int((pred == y_test).sum())
        total = int(len(y_test))
        acc = 100.0 * correct / total if total else 0.0
        print(f"SNR {snr:3d} dB -> {group_name(group_id):>10} head | Acc {acc:6.2f}%")
        test_rows.append(
            {
                "experiment": args.experiment_name,
                "snr": snr,
                "head": group_name(group_id),
                "modulation": "ALL",
                "label": "ALL",
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "samples_per_head": args.samples_per_head,
                "epochs": args.epochs,
                "checkpoint": args.best_path,
            }
        )
        for true_label, modulation in enumerate(modulations):
            mask = y_test == true_label
            mod_total = int(mask.sum())
            mod_correct = int((pred[mask] == y_test[mask]).sum())
            test_rows.append(
                {
                    "experiment": args.experiment_name,
                    "snr": snr,
                    "head": group_name(group_id),
                    "modulation": modulation,
                    "label": true_label,
                    "accuracy": 100.0 * mod_correct / mod_total if mod_total else 0.0,
                    "correct": mod_correct,
                    "total": mod_total,
                    "samples_per_head": args.samples_per_head,
                    "epochs": args.epochs,
                    "checkpoint": args.best_path,
                }
            )
            for pred_label, pred_mod in enumerate(modulations):
                count = int(((pred == pred_label) & mask).sum())
                confusion_rows.append(
                    {
                        "experiment": args.experiment_name,
                        "snr": snr,
                        "head": group_name(group_id),
                        "true_label": true_label,
                        "true_modulation": modulation,
                        "pred_label": pred_label,
                        "pred_modulation": pred_mod,
                        "count": count,
                        "true_total": mod_total,
                        "row_percent": 100.0 * count / mod_total if mod_total else 0.0,
                        "samples_per_head": args.samples_per_head,
                        "epochs": args.epochs,
                        "checkpoint": args.best_path,
                    }
                )

    write_csv(
        args.results_csv,
        ["experiment", "snr", "head", "modulation", "label", "accuracy", "correct", "total", "samples_per_head", "epochs", "checkpoint"],
        test_rows,
    )
    write_csv(
        args.confusion_csv,
        ["experiment", "snr", "head", "true_label", "true_modulation", "pred_label", "pred_modulation", "count", "true_total", "row_percent", "samples_per_head", "epochs", "checkpoint"],
        confusion_rows,
    )
    summary = {
        "experiment": args.experiment_name,
        "model": "SSL encoder + 3 SNR heads, equal samples per head",
        "samples_per_head": args.samples_per_head,
        "total_labeled": int(len(x_small)),
        "best_val_accuracy": best_val_acc,
        "best_checkpoint": args.best_path,
        "last_checkpoint": args.last_path,
        "results_csv": args.results_csv,
        "confusion_csv": args.confusion_csv,
        "log_csv": args.log_csv,
        "sample_summary": sample_summary,
    }
    write_json(args.summary_json, summary)
    write_json(args.metrics_json, build_metrics_payload(args, modulations, train_log_rows, confusion_rows, best_val_acc, sample_summary))


if __name__ == "__main__":
    main()
