# Benchmark Results

This directory contains results from `benchmarks/benchmark_all_models.py`.

Run the benchmark with:
```bash
# Full benchmark (80 epochs, 300 samples/class, 3 runs each)
python benchmarks/benchmark_all_models.py

# Quick smoke-test (5 epochs, 50 samples/class)
python benchmarks/benchmark_all_models.py --quick
```

---

## Expected Results (Synthetic Dataset)

Results below are indicative targets based on the dataset characteristics.
Run `benchmark_all_models.py` to produce actual numbers in `benchmark_results.csv`.

### Inverter Fault Dataset (9 classes, 6 channels)

| Model | Accuracy | Macro F1 | Parameters |
|---|---|---|---|
| **1D CNN (Residual)** | ~97–99% | ~0.97 | ~1.2M |
| **Spectrogram CNN (ResNet-18)** | ~96–98% | ~0.96 | ~11M |
| **Transformer (PatchTST-style)** | ~95–97% | ~0.95 | ~800K |
| **BiLSTM + Attention** | ~94–96% | ~0.94 | ~1.8M |

### Motor Drive Fault Dataset (5 classes, 3 channels)

| Model | Accuracy | Macro F1 | Parameters |
|---|---|---|---|
| **1D CNN (Residual)** | ~98–99% | ~0.98 | ~1.1M |
| **Spectrogram CNN (ResNet-18)** | ~97–99% | ~0.97 | ~11M |
| **Transformer (PatchTST-style)** | ~96–98% | ~0.96 | ~750K |
| **BiLSTM + Attention** | ~95–97% | ~0.95 | ~1.7M |

### Autoencoder (Unsupervised — Anomaly Detection)

Evaluated on overtemperature vs. healthy:

| Metric | Value |
|---|---|
| AUROC | ~0.90–0.95 |
| Threshold (95th pct) | dataset-dependent |

---

> **Note:** Actual results may differ due to random seed, hardware, and dataset size.
> Run the benchmark script to generate reproducible results on your machine.
