# Denoising SSL Baseline

This folder contains the denoising SSL result used as a negative baseline.

Included files:

- `ssl_metrics.json`: evaluation metrics for the denoising SSL model.
- `RESULT_ANALYSIS.md`: written analysis of the denoising experiment.

The denoising condition used an MSE-style waveform reconstruction objective. In the current task, this objective did not produce discriminative modulation features and remained close to chance level across SNRs. This result is interpreted as evidence that naive waveform denoising is not well aligned with the modulation classification objective.
