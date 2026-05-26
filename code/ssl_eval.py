# contrastive SSL로 사전학습한 encoder를 불러와 low-label fine-tuning과 평가를 수행하는 코드.

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


class SimpleAug:
    def __call__(self, x):
        x_aug = x.clone().float()
        if np.random.rand() < 0.5:
            x_aug *= np.random.uniform(0.95, 1.05)
        if np.random.rand() < 0.5:
            x_aug += torch.randn_like(x_aug) * 0.001
        return x_aug


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
            class_idx = indices[y_label[indices] == c]
        else:
            class_idx = indices[(y_label[indices] == c) & (snr_label[indices] == snr)]
        if len(class_idx) == 0:
            continue
        if label_ratio is not None:
            sample_count = max(1, int(len(class_idx) * label_ratio))
        else:
            sample_count = min(num_per_class, len(class_idx))
        selected_idx = class_idx[:sample_count]
        x_list.append(x_data[selected_idx])
        y_list.append(y_label[selected_idx])

    return np.concatenate(x_list), np.concatenate(y_list)


def load_train_val_data(args):
    split_paths = [
        args.train_x,
        args.train_y,
        args.val_x,
        args.val_y,
    ]
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
            class_idx = np.where(y_label == c)[0]
        else:
            class_idx = np.where((y_label == c) & (snr_label == snr))[0]
        if len(class_idx) == 0:
            continue
        np.random.shuffle(class_idx)
        val_count = max(1, int(len(class_idx) * 0.2))
        val_idx.extend(class_idx[:val_count])
        train_idx.extend(class_idx[val_count:])
    return (
        x_data[train_idx],
        y_label[train_idx],
        None if snr_label is None else snr_label[train_idx],
        x_data[val_idx],
        y_label[val_idx],
        None if snr_label is None else snr_label[val_idx],
    )


def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs.to(device))
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum().item()
    return 100 * correct / total


def parse_csv_list(value, item_type=str):
    return [item_type(item.strip()) for item in value.split(",") if item.strip()]


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


def write_train_log_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "epoch", "train_loss", "val_accuracy", "best_val_accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_test_results_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
        )
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def build_metrics_payload(args, modulations, train_log_rows, confusion_rows, best_val_acc, best_path, last_path):
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
    snr_acc = {}
    for snr_key, snr_matrix in snr_matrices.items():
        snr_total = int(snr_matrix.sum())
        snr_acc[snr_key] = float(np.trace(snr_matrix) / snr_total) if snr_total else 0.0

    mod_acc = {}
    for label in range(args.num_classes):
        label_total = int(matrix[label].sum())
        name = modulations[label] if label < len(modulations) else f"class_{label}"
        mod_acc[name] = float(matrix[label, label] / label_total) if label_total else 0.0

    history = [
        {
            "epoch": row["epoch"],
            "train_loss": row["train_loss"],
            "val_acc": row["val_accuracy"] / 100,
        }
        for row in train_log_rows
    ]

    return {
        "best_val_acc": best_val_acc / 100,
        "test_acc": test_acc,
        "snr_acc": snr_acc,
        "mod_acc": mod_acc,
        "confusion_matrix": matrix.tolist(),
        "snr_confusion_matrix": {key: value.tolist() for key, value in snr_matrices.items()},
        "history": history,
        "config": {
            "experiment": args.experiment_name,
            "model": "SSL fine-tuning",
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
            "pretrained_path": args.pretrained_path,
            "modulations": modulations,
            "num_classes": args.num_classes,
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
            "results_csv": args.results_csv,
            "confusion_csv": args.confusion_csv,
            "log_csv": args.log_csv,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SSL encoder for modulation classification.")
    parser.add_argument("--x-data", default="X_data_4096.npy")
    parser.add_argument("--y-label", default="Y_label_4096.npy")
    parser.add_argument("--snr-label", default="SNR_label_4096.npy")
    parser.add_argument("--train-x", default="X_train_4096.npy")
    parser.add_argument("--train-y", default="Y_train_4096.npy")
    parser.add_argument("--train-snr", default="SNR_train_4096.npy")
    parser.add_argument("--val-x", default="X_val_4096.npy")
    parser.add_argument("--val-y", default="Y_val_4096.npy")
    parser.add_argument("--val-snr", default="SNR_val_4096.npy")
    parser.add_argument("--test-dir", default="test_data_snr")
    parser.add_argument("--test-snrs", default="-10,-5,0,5,10,15,20")
    parser.add_argument("--pretrained-path", default="ssl_encoder_cuda_epoch_20.pth")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--modulations", default="BPSK,QPSK,8PSK,16QAM,64QAM")
    parser.add_argument("--num-per-class", type=int, default=200)
    parser.add_argument("--label-ratio", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--classifier-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--best-path", default="best_ssl_finetune.pth")
    parser.add_argument("--last-path", default="last_ssl_finetune.pth")
    parser.add_argument("--results-csv", default="ssl_finetune_results.csv")
    parser.add_argument("--confusion-csv", default="ssl_finetune_confusion.csv")
    parser.add_argument("--log-csv", default="ssl_finetune_train_log.csv")
    parser.add_argument("--summary-json", default="ssl_finetune_summary.json")
    parser.add_argument("--metrics-json", default="ssl_metrics.json")
    parser.add_argument("--experiment-name", default="SSL")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


args = parse_args()
modulations = parse_csv_list(args.modulations, str)
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
best_path = args.best_path
last_path = args.last_path

X_train_raw, Y_train_raw, SNR_train_raw, X_val_raw, Y_val_raw, SNR_val_raw = load_train_val_data(args)
X_train_small, Y_train_small = stratified_sample(
    X_train_raw,
    Y_train_raw,
    num_classes=args.num_classes,
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

base_encoder = SSLResNet().to(device)
base_encoder.load_state_dict(
    torch.load(args.pretrained_path, map_location=device, weights_only=True)
)
model = SSLClassifier(base_encoder, num_classes=args.num_classes).to(device)

optimizer = optim.Adam(
    [
        {"params": model.encoder.parameters(), "lr": args.encoder_lr},
        {"params": model.fc.parameters(), "lr": args.classifier_lr},
    ],
    weight_decay=args.weight_decay,
)
criterion = nn.CrossEntropyLoss()
aug = SimpleAug()

print(
    f"SSL fine-tuning start: {len(X_train_small)} labeled samples, "
    f"{args.num_classes} classes, label_ratio={args.label_ratio}, "
    f"num_per_class={args.num_per_class}, pretrained={args.pretrained_path}"
)

best_val_acc = 0.0
train_log_rows = []
for epoch in range(args.epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        inputs_aug = torch.stack([aug(x) for x in inputs]).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs_aug)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    val_acc = evaluate_accuracy(model, val_loader, device)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_path)
    train_log_rows.append(
        {
            "experiment": args.experiment_name,
            "epoch": epoch + 1,
            "train_loss": epoch_loss / len(train_loader),
            "val_accuracy": val_acc,
            "best_val_accuracy": best_val_acc,
        }
    )

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1:04d} | Avg Loss: {epoch_loss / len(train_loader):.4f} "
            f"| Val Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}%"
        )
    if device == "cuda":
        torch.cuda.empty_cache()

torch.save(model.state_dict(), last_path)
write_train_log_csv(args.log_csv, train_log_rows)

print("\n" + "=" * 50)
print(f"SSL fine-tuning final result using best val checkpoint ({best_val_acc:.2f}%)")
print("=" * 50)

model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
model.eval()

test_rows = []
confusion_rows = []
for snr in [int(item.strip()) for item in args.test_snrs.split(",") if item.strip()]:
    x_path = os.path.join(args.test_dir, f"X_test_snr_4096_{snr}.npy")
    y_path = os.path.join(args.test_dir, f"Y_test_snr_4096_{snr}.npy")

    if not os.path.exists(x_path):
        continue

    X_test_raw = np.load(x_path).astype(np.float32)
    Y_test_raw = np.load(y_path).astype(np.int64)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test_raw), torch.from_numpy(Y_test_raw)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    pred, labels = evaluate_predictions(model, test_loader, device)
    correct = int((pred == labels).sum())
    total = int(len(labels))
    acc = 100 * correct / total
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
            "epochs": args.epochs,
            "checkpoint": best_path,
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
                "accuracy": 100 * mod_correct / mod_total,
                "correct": mod_correct,
                "total": mod_total,
                "label_ratio": args.label_ratio,
                "num_per_class": args.num_per_class,
                "epochs": args.epochs,
                "checkpoint": best_path,
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
                    "checkpoint": best_path,
                }
            )

write_test_results_csv(args.results_csv, test_rows)
write_confusion_csv(args.confusion_csv, confusion_rows)
write_summary_json(
    args.summary_json,
    {
        "experiment": args.experiment_name,
        "model": "SSL fine-tuning",
        "num_classes": args.num_classes,
        "modulations": modulations,
        "label_ratio": args.label_ratio,
        "num_per_class": args.num_per_class,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "encoder_lr": args.encoder_lr,
        "classifier_lr": args.classifier_lr,
        "weight_decay": args.weight_decay,
        "pretrained_path": args.pretrained_path,
        "best_val_accuracy": best_val_acc,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "results_csv": args.results_csv,
        "confusion_csv": args.confusion_csv,
        "log_csv": args.log_csv,
    },
)
write_summary_json(
    args.metrics_json,
    build_metrics_payload(args, modulations, train_log_rows, confusion_rows, best_val_acc, best_path, last_path),
)
print(f"Saved results: {args.results_csv}")
print(f"Saved confusion matrix: {args.confusion_csv}")
print(f"Saved train log: {args.log_csv}")
print(f"Saved summary: {args.summary_json}")
print(f"Saved metrics JSON: {args.metrics_json}")
