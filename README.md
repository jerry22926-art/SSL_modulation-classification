# Self-Supervised Modulation Classification under Varying SNR

This repository bundle contains the code, paper figures, and compact result files for low-label modulation classification under varying SNR.

The bundle is designed for paper linking: it keeps the evidence needed to inspect the experiments, while excluding large generated data arrays, model checkpoints, Colab notebooks, and CSV dumps.

## Experiment Setting

| Item | Setting |
|---|---|
| Task | 5-class modulation classification |
| Modulations | BPSK, QPSK, 8PSK, 16QAM, 64QAM |
| Input | I/Q signal, shape `(2, 4096)` |
| Channel | Synthetic AWGN SNR sweep |
| Default training SNRs | `0, 5, 10, 15, 20` dB |
| Low-SNR exposure SNRs | default training SNRs plus `-10, -5` dB |
| Per-head SSL train groups | severe `[-10, -5]`, transition `[0, 5]`, high `[10, 15, 20]` |
| Main label ratio | `0.005` |
| Labeled training scale | baseline `0.005` uses about 200 labeled samples per modulation class; per-head SSL uses 200 labeled samples per head, 600 total |
| Dense test SNRs | `-10, -5, 0, 1, 2, 3, 4, 5, 10, 15, 20` dB |
| Test samples | 500 samples per modulation per SNR, 2500 total per SNR |
| Chance level | 20% |
| Main methods | SL, contrastive SSL, low-SNR exposure SSL, per-head SSL, denoising SSL |

## Model and Training Method

### Backbone

All main SL and SSL experiments use the same 1D residual CNN backbone for I/Q signals.

```text
Input I/Q signal
    shape: (2, 4096)
        |
        v
Per-sample std normalization
        |
        v
ResBlock 1D: 2 -> 64    + MaxPool1d(2)
        |
        v
ResBlock 1D: 64 -> 128  + MaxPool1d(2)
        |
        v
ResBlock 1D: 128 -> 256 + MaxPool1d(2)
        |
        v
ResBlock 1D: 256 -> 512 + MaxPool1d(2)
        |
        v
Flattened encoder feature
```

- Input: two I/Q channels with sequence length 4096.
- Encoder: four residual 1D convolution blocks with channel sizes `2 -> 64 -> 128 -> 256 -> 512`.
- Downsampling: max pooling after each residual block.
- Feature vector: flattened final convolution feature map.
- SL classifier: a linear classification head over the encoder feature.
- SSL projection head: `Linear(512*256, 512) -> ReLU -> Linear(512, 128)`.

The encoder normalizes each sample by its per-sample temporal standard deviation before the convolution blocks.

### Contrastive SSL

SSL pretraining uses a SimCLR-style NT-Xent objective. For two augmented views of the same signal, `(i, j)` is treated as a positive pair and other views in the batch are treated as negatives.

```text
Original I/Q signal
      |                         |
      v                         v
 RF augmentation view 1   RF augmentation view 2
      |                         |
      v                         v
 Shared 1D ResNet encoder Shared 1D ResNet encoder
      |                         |
      v                         v
 Projection head z_i      Projection head z_j
      \_________________________/
                 |
                 v
          NT-Xent contrastive loss
```

The loss is:

```text
L_i = -log exp(sim(z_i, z_j) / tau) / sum_{k != i} exp(sim(z_i, z_k) / tau)
```

where `sim` is cosine similarity and `tau = 0.5`.

The SSL augmentation pipeline applies RF-style perturbations such as additive noise, amplitude scaling, phase rotation, and circular time shift. Fine-tuning then uses the pretrained encoder with a supervised cross-entropy classifier.

### Per-Head SSL

The per-head SSL experiment keeps one shared SSL encoder and separates the classifier into three SNR-regime heads:

```text
Input I/Q signal
        |
        v
Shared SSL encoder
        |
        v
Encoder feature
        |
        +------------------> Severe head      (-10, -5 dB)
        |
        +------------------> Transition head  (0-5 dB)
        |
        +------------------> High head        (10, 15, 20 dB)
```

- severe head: `-10, -5` dB
- transition head: `0, 5` dB during training, evaluated densely on `0-5` dB
- high head: `10, 15, 20` dB

This design tests whether a single encoder can benefit from SNR-specific decision boundaries in the classifier.

## Folder Structure

- `code/`: Python source files for signal generation, SL/SSL training, dense SNR evaluation, denoising SSL, and SNR-aware/per-head classifiers.
- `results/baseline_dense/`: SL and contrastive SSL baseline dense SNR evaluation.
- `results/low_snr_exposure/`: Low-SNR exposure ablation.
- `results/per_head_ssl/`: SNR-regime per-head SSL experiment.
- `results/denoising_ssl/`: Denoising SSL negative baseline.
- `results/figures/`: Paper-ready figures.

## Main Figure

The main paper figure is:

- `results/figures/figure_1_main_snr_comparison.png`
- `results/figures/figure_1_main_snr_comparison.pdf`
- `results/figures/figure_1_main_snr_comparison.svg`

It compares five conditions at label ratio `0.005`:

- SL baseline
- contrastive SSL baseline
- SSL with low-SNR information added to both pretraining and fine-tuning
- SNR-regime per-head SSL
- denoising SSL

## Result Files

### Baseline Dense SNR

- `results/baseline_dense/sl_dense_eval_metrics.json`
- `results/baseline_dense/ssl_dense_eval_metrics.json`
- `results/baseline_dense/README.md`

This compares the supervised baseline and contrastive SSL baseline on the dense SNR test set.

### Low-SNR Exposure

- `results/low_snr_exposure/low_snr_exposure_metrics.json`
- `results/low_snr_exposure/README.md`

This ablation tests whether adding `-10` and `-5` dB samples to pretraining and/or fine-tuning improves low-SNR behavior. The result is mainly negative: simple exposure to low-SNR data did not meaningfully solve severe low-SNR performance.

### Per-Head SSL

- `results/per_head_ssl/per_head_multihead_metrics.json`
- `results/per_head_ssl/per_head_multihead_summary.json`
- `results/per_head_ssl/README.md`

This experiment uses a shared SSL encoder with separate classifier heads for SNR regimes. It improves the transition and high-SNR regimes more clearly than severe low-SNR.

### Denoising SSL

- `results/denoising_ssl/ssl_metrics.json`
- `results/denoising_ssl/README.md`

This is a negative baseline using MSE-style denoising pretraining. It stayed close to chance level, suggesting that naive waveform reconstruction is not well aligned with modulation-discriminative representation learning.

## Main Takeaways

1. Contrastive SSL improves label efficiency mainly when enough class-discriminative signal remains.
2. Severe low-SNR conditions, especially `-10` and `-5` dB, remain near chance level across most variants.
3. The practical transition region appears around `1-5` dB, where SSL and per-head SSL gains become visible.
4. Low-SNR exposure alone is not sufficient; simply showing more low-SNR samples does not recover the lost modulation cues.
5. Per-head SSL is a useful SNR-aware extension because it lets different classifier heads specialize to different SNR regimes.
6. Denoising SSL with a simple MSE reconstruction objective is not a competitive baseline for this task.

## Excluded From This Bundle

The following files are intentionally excluded:

- raw/generated NumPy arrays (`*.npy`, `*.npz`)
- model checkpoints (`*.pth`, `*.pt`, `*.ckpt`)
- Colab notebooks
- large zip archives
- CSV dumps and training logs
- Python cache files

This keeps the paper link focused on readable code, compact JSON metrics, figures, and method/result notes.
