# Baseline Dense SNR Evaluation

This folder summarizes the main dense SNR evaluation for the low-label `0.005` setting.

Included conditions:

- `sl_dense_eval_metrics.json`: supervised learning baseline.
- `ssl_dense_eval_metrics.json`: contrastive SSL encoder followed by supervised fine-tuning.
- `dense_eval_accuracy_curve.png`: dense SNR accuracy curve for the baseline evaluation.
- `RESULT_ANALYSIS.md`: written interpretation of the SL vs SSL dense SNR results.

The key observation is that SSL does not improve all SNR regimes equally. Severe low-SNR remains close to chance level, while transition and high-SNR regimes show clear SSL gains.
