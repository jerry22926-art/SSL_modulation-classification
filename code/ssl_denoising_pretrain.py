# 잡음이 섞인 I/Q 신호를 복원하는 MSE denoising SSL pretraining을 수행하는 코드.

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class DenoisingDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


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

    def forward_features(self, x):
        std = torch.std(x, dim=-1, keepdim=True) + 1e-8
        x = x / std
        x = self.pool(self.layer1(x))
        x = self.pool(self.layer2(x))
        x = self.pool(self.layer3(x))
        x = self.pool(self.layer4(x))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        return self.projection_head(x)


class DenoisingDecoder(nn.Module):
    def __init__(self):
        super(DenoisingDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(32, 2, kernel_size=3, padding=1),
        )

    def forward(self, features):
        return self.net(features)


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = SSLResNet()
        self.decoder = DenoisingDecoder()

    def forward(self, x_noisy):
        features = self.encoder.forward_features(x_noisy)
        return self.decoder(features)


def write_log_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Denoising SSL pretraining for synthetic I/Q data.")
    parser.add_argument("--x-data", default="X_train_4096.npy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--save-dir", default=".")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--log-csv", default="ssl_denoising_pretrain_log.csv")
    parser.add_argument("--summary-json", default="ssl_denoising_pretrain_summary.json")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


args = parse_args()
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

os.makedirs(args.save_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"

dataset = DenoisingDataset(args.x_data)
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=use_amp,
)
if len(loader) == 0:
    raise ValueError("Dataset is smaller than batch_size with drop_last=True.")

model = DenoisingAutoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
log_rows = []

print("Denoising SSL pretraining start...")
print(f"x_data={args.x_data}, noise_std={args.noise_std}, device={device}")

for epoch in range(args.epochs):
    model.train()
    total_loss = 0.0
    for x in loader:
        x = x.to(device)
        x_noisy = x + torch.randn_like(x) * args.noise_std
        target_std = torch.std(x, dim=-1, keepdim=True) + 1e-8
        target = x / target_std

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            recon = model(x_noisy)
            loss = F.mse_loss(recon, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    if device == "cuda":
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(loader)
    log_rows.append({"epoch": epoch + 1, "loss": avg_loss})
    print(f"Epoch {epoch + 1:03d} | MSE Loss: {avg_loss:.6f}")

    if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
        save_path = os.path.join(args.save_dir, f"ssl_denoising_encoder_cuda_epoch_{epoch + 1}.pth")
        torch.save(model.encoder.state_dict(), save_path)

log_path = os.path.join(args.save_dir, args.log_csv)
summary_path = os.path.join(args.save_dir, args.summary_json)
write_log_csv(log_path, log_rows)
write_json(
    summary_path,
    {
        "x_data": args.x_data,
        "objective": "denoising_mse",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "noise_std": args.noise_std,
        "save_dir": args.save_dir,
        "save_every": args.save_every,
        "device": device,
        "final_loss": log_rows[-1]["loss"] if log_rows else None,
        "checkpoint_pattern": "ssl_denoising_encoder_cuda_epoch_*.pth",
        "compatible_with": "ssl_eval.py SSLResNet state_dict",
    },
)

print("Denoising SSL pretraining done!")
print(f"Saved log: {log_path}")
print(f"Saved summary: {summary_path}")
