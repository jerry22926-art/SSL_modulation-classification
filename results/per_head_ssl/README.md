# SSL Per-Head SNR Multi-Head Low-SNR Train Result

Source zip:
`per_head_snr_multihead_lowtrain_results.zip`

## Setup

- Encoder: contrastive SSL encoder from `ssl_pretrain.zip`
- Train SNRs: `-10,-5,0,5,10,15,20`
- Test SNRs: `-10,-5,0,1,2,3,4,5,10,15,20`
- Heads:
  - low: train `-10,-5`, test `-10,-5`
  - transition: train `0,5`, test `0,1,2,3,4,5`
  - high: train `10,15,20`, test `10,15,20`
- Labeled samples: `200` per head, `600` total
- Test samples: `2500` per SNR

## Accuracy vs Baselines

| SNR | SL 0.005 | SSL 0.005 | Per-head | Gain vs SSL |
|---:|---:|---:|---:|---:|
| -10 | 19.52 | 20.60 | 18.44 | -2.16 |
| -5 | 20.60 | 18.96 | 21.84 | +2.88 |
| 0 | 21.68 | 24.24 | 28.32 | +4.08 |
| 1 | 22.64 | 28.92 | 36.80 | +7.88 |
| 2 | 23.12 | 36.64 | 44.36 | +7.72 |
| 3 | 25.16 | 42.76 | 46.64 | +3.88 |
| 4 | 28.56 | 49.76 | 52.64 | +2.88 |
| 5 | 30.08 | 56.24 | 55.36 | -0.88 |
| 10 | 40.92 | 71.12 | 71.92 | +0.80 |
| 15 | 43.40 | 78.36 | 83.24 | +4.88 |
| 20 | 44.24 | 82.96 | 88.48 | +5.52 |

## Summary

- Overall gain vs SSL baseline: `+3.41` percentage points.
- Severe low SNR `-10,-5`: essentially unchanged, `+0.36` points on average.
- Transition SNR `0..5`: improved by `+4.26` points on average.
- High SNR `10,15,20`: improved by `+3.73` points on average.
- The strongest gains are at `1 dB` and `2 dB`, where per-head multi-head improves over SSL by `+7.88` and `+7.72` points.

