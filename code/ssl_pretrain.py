# SimCLR-style contrastive learning으로 I/Q encoder를 self-supervised pretraining하는 코드.

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


class RFDataAugmentation:
    def __init__(self, p_awgn=0.8, p_scale=0.8, p_phase=0.6):
        self.p_awgn = p_awgn
        self.p_scale = p_scale
        self.p_phase = p_phase

    def __call__(self, x):
        x_aug = x.clone().float()

        if np.random.rand() < self.p_scale:
            x_aug = x_aug * np.random.uniform(0.5, 1.5)

        if np.random.rand() < self.p_phase:
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            i_comp = x_aug[0] * cos_a - x_aug[1] * sin_a
            q_comp = x_aug[0] * sin_a + x_aug[1] * cos_a
            x_aug = torch.stack([i_comp, q_comp])

        if np.random.rand() < self.p_awgn:
            x_aug = x_aug + torch.randn_like(x_aug) * 0.01

        return x_aug


class SSLDataset(Dataset):
    def __init__(self, data_path, augmentation):
        self.data = np.load(data_path).astype(np.float32)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        return self.augmentation(x), self.augmentation(x)


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


def nt_xent_loss(z1, z2, temperature=0.5):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix.masked_fill_(mask, -3e4)
    pos_indices = torch.arange(batch_size, device=z.device)
    labels = torch.cat([pos_indices + batch_size, pos_indices], dim=0)
    return F.cross_entropy(sim_matrix, labels)


def parse_args():
    parser = argparse.ArgumentParser(description="SSL pretraining for synthetic I/Q data.")
    parser.add_argument("--x-data", default="X_data_4096.npy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--save-dir", default=".")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--log-csv", default="ssl_pretrain_log.csv")
    parser.add_argument("--summary-json", default="ssl_pretrain_summary.json")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def write_log_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


args = parse_args()
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

os.makedirs(args.save_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"

dataset = SSLDataset(args.x_data, augmentation=RFDataAugmentation())
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

model = SSLResNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
log_rows = []

print("SSL Pre-training start...")
for epoch in range(args.epochs):
    model.train()
    total_loss = 0.0
    for x1, x2 in loader:
        x1, x2 = x1.to(device), x2.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            z1, z2 = model(x1), model(x2)
            loss = nt_xent_loss(z1, z2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    if device == "cuda":
        torch.cuda.empty_cache()
    avg_loss = total_loss / len(loader)
    log_rows.append({"epoch": epoch + 1, "loss": avg_loss})
    print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f}")

    if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
        save_path = os.path.join(args.save_dir, f"ssl_encoder_cuda_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)

log_path = os.path.join(args.save_dir, args.log_csv)
summary_path = os.path.join(args.save_dir, args.summary_json)
write_log_csv(log_path, log_rows)
write_json(
    summary_path,
    {
        "x_data": args.x_data,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "save_dir": args.save_dir,
        "save_every": args.save_every,
        "device": device,
        "final_loss": log_rows[-1]["loss"] if log_rows else None,
    },
)

print("SSL Pre-training done!")
print(f"Saved log: {log_path}")
print(f"Saved summary: {summary_path}")
