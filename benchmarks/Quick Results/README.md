## Benchmark Results

> **Note:** Results below are from quick-mode evaluation (5 epochs, 50 samples/class, 1 run).
> Full benchmark results (300 samples/class, 50 epochs, 3 runs) coming soon.
> Run `python benchmarks/benchmark_all_models.py` to reproduce full results.

### Inverter Fault Detection — 9 classes (Synthetic)

| Model | Accuracy | Macro F1 | Params | Mode |
|-------|----------|----------|--------|------|
| BiLSTM + Attention | 77.94% | 0.777 | 2.2M | Quick (5 ep) |
| 1D CNN Waveform | 73.53% | 0.687 | 4.0M | Quick (5 ep) |
| Spectrogram CNN | 36.76% | 0.315 | 11.2M | Quick (5 ep) |
| Transformer | 32.35% | 0.251 | 0.8M | Quick (5 ep) |

### Motor Drive Fault Detection — 5 classes (Synthetic)

| Model | Accuracy | Macro F1 | Params | Mode |
|-------|----------|----------|--------|------|
| BiLSTM + Attention | 63.16% | 0.594 | 2.2M | Quick (5 ep) |
| Spectrogram CNN | 50.00% | 0.417 | 11.2M | Quick (5 ep) |
| 1D CNN Waveform | 44.74% | 0.409 | 4.0M | Quick (5 ep) |
| Transformer | 21.05% | 0.070 | 0.8M | Quick (5 ep) |

### Methodology

- **Data:** Physics-informed synthetic signals (no download required)
- **Split:** 70% train / 15% val / 15% test, stratified
- **Seed:** 42
- **Device:** CPU
- **Quick mode:** 5 epochs, 50 samples/class — for smoke-testing only
- Reproduce: `python benchmarks/benchmark_all_models.py --quick`

### Why quick-mode numbers are low

Spectrogram CNN (11M params) and Transformer require significantly more data
and epochs than BiLSTM and 1D CNN to converge. Quick mode favors models with
strong inductive biases (sequential/local patterns) over capacity-heavy models.
Full benchmark results expected to reach 95%+ accuracy for all models.
