# Low-SNR Exposure Ablation

This folder contains the low-SNR exposure ablation for the `0.005` label-ratio setting.

## Conditions

- `D_low_pretrain_ssl`: SSL pretraining includes `-10` and `-5` dB, fine-tuning uses the original SNR set.
- `E_low_finetune_ssl`: SSL pretraining uses the original SNR set, fine-tuning includes `-10` and `-5` dB.
- `F_low_both_ssl`: both SSL pretraining and fine-tuning include `-10` and `-5` dB.

## Test Setup

- Test SNRs: `-10, -5, 0, 1, 2, 3, 4, 5, 10, 15, 20`
- Samples per modulation per SNR: `500`
- Total samples per SNR: `2500`
- Label ratio: `0.005`

## Files

- `low_snr_exposure_metrics.json`: compact metrics for the low-SNR exposure variants.
- `ssl_0p005_low_snr_variants_dense_eval_curve.png`: dense SNR curve for the exposure variants.

## Interpretation

Low-SNR exposure did not meaningfully improve severe low-SNR accuracy. Even when low-SNR samples were added to both pretraining and fine-tuning, `-10` and `-5` dB stayed close to chance level. This condition is therefore best interpreted as a negative ablation: simple exposure to low-SNR samples is not enough to recover modulation-discriminative cues.
