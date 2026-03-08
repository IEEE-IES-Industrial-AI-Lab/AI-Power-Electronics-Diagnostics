# AI Power Electronics Diagnostics

> A research framework for AI-based fault detection in industrial power electronics systems, developed by the **IEEE IES Industrial AI Lab**.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IEEE IES](https://img.shields.io/badge/org-IEEE%20IES-blue.svg)](https://www.ieee-ies.org/)

---

## Overview

Industrial power electronics systems — inverters, motor drives, and converters — are critical components in manufacturing, renewable energy, and transportation. Their failure causes costly unplanned downtime. This repository provides a complete AI pipeline for **early fault detection from electrical signals** (voltage waveforms, current signals, harmonic spectrum).

### What this repo covers

| Domain | Fault Types | Signals |
|---|---|---|
| **3-Phase VSI Inverter** | Open-circuit IGBT (T1–T6), Short-circuit, DC undervoltage | Va, Vb, Vc, Ia, Ib, Ic |
| **PMSM Motor Drive** | Phase loss, Inter-turn short circuit, Bearing fault, Overtemperature | Ia, Ib, Ic |

---

## Repository Structure

```
ai-power-electronics-diagnostics/
│
├── datasets/
│   ├── synthetic/           # Physics-informed signal generators (no download needed)
│   │   ├── inverter_fault_sim.py
│   │   ├── motor_drive_sim.py
│   │   └── fault_injector.py
│   ├── loaders/             # Real dataset loaders
│   │   ├── base_loader.py
│   │   └── motor_temp_loader.py
│   └── download_scripts/    # Kaggle API helpers
│
├── signal_processing/       # FFT, STFT, wavelet, harmonic analysis
│   ├── fft_analysis.py
│   ├── stft_spectrogram.py
│   ├── wavelet_features.py
│   ├── harmonic_analysis.py
│   └── feature_extractor.py
│
├── models/                  # 5 PyTorch model architectures
│   ├── cnn_waveform_classifier.py   # 1D Residual CNN
│   ├── spectrogram_cnn.py           # ResNet-18 on STFT spectrograms
│   ├── transformer_signal.py        # Patch-based Transformer
│   ├── lstm_classifier.py           # BiLSTM + Attention
│   └── autoencoder_anomaly.py       # Unsupervised 1D Autoencoder
│
├── fault_detection/         # End-to-end detection pipelines
│   ├── switch_fault_detector.py
│   ├── harmonic_fault_detector.py
│   └── thermal_fault_detector.py
│
├── training/
│   ├── train.py             # CLI training script
│   ├── evaluate.py          # Evaluation + metrics report
│   ├── config.yaml          # All hyperparameters
│   └── utils.py             # Early stopping, checkpointing, schedulers
│
├── visualization/
│   ├── waveform_plots.py
│   ├── spectrogram_plots.py
│   └── fault_dashboard.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_signal_processing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_fault_detection_demo.ipynb
│
└── benchmarks/
    ├── benchmark_all_models.py
    └── results/
```

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/IEEE-IES-Industrial-AI-Lab/AI-Power-Electronics-Diagnostics.git
cd AI-Power-Electronics-Diagnostics
pip install -r requirements.txt
```

### 2. Train a model (synthetic data, no download needed)

```bash
# Train 1D CNN on inverter fault data
python training/train.py --model cnn_waveform --dataset synthetic --fault_domain inverter

# Train Transformer on motor drive data
python training/train.py --model transformer --dataset synthetic --fault_domain motor

# Train autoencoder for unsupervised anomaly detection
python training/train.py --model autoencoder --dataset synthetic
```

### 3. Evaluate

```bash
python training/evaluate.py \
    --checkpoint training/checkpoints/<experiment_name>/best.pt \
    --model cnn_waveform \
    --dataset synthetic
```

### 4. Run full benchmark

```bash
# Quick smoke-test (~2 min)
python benchmarks/benchmark_all_models.py --quick

# Full benchmark (~1–4 hours depending on hardware)
python benchmarks/benchmark_all_models.py
```

### 5. Explore notebooks

```bash
jupyter notebook notebooks/
```

---

## Models

### 1D CNN — `cnn_waveform_classifier.py`

Residual 1D convolutional network operating directly on raw waveform windows.

```
Input (B, C, T) → Stem → 4× ResidualBlock1D → GlobalAvgPool → Classifier
```

- Fast training, strong baseline
- ~1.2M parameters for 6-channel input

### Spectrogram CNN — `spectrogram_cnn.py`

ResNet-18 applied to STFT spectrograms. Captures time-frequency fault signatures.

```
Input (B, C, H, W) → Flexible Stem → 4× ResNet Stages → GAP → Classifier
```

- Best for transient fault detection
- ~11M parameters

### Transformer — `transformer_signal.py`

Patch-based Transformer encoder, inspired by PatchTST (Nie et al., 2022).
Each channel is patched independently with shared weights.

```
Input (B, C, T) → Patch Embedding → Positional Encoding → 4× Encoder Blocks → CLS → Classifier
```

- Captures long-range temporal dependencies
- ~800K parameters

### BiLSTM + Attention — `lstm_classifier.py`

2-layer bidirectional LSTM with learnable additive attention pooling.

```
Input (B, C, T) → BiLSTM × 2 → Additive Attention → LayerNorm → Classifier
```

- Sequential fault pattern recognition
- Returns attention weights for interpretability

### Autoencoder — `autoencoder_anomaly.py`

Unsupervised 1D convolutional autoencoder. Trained only on healthy signals.
Anomaly score = reconstruction MSE.

```
Encoder: Conv1d × 4 (strided) → Bottleneck
Decoder: ConvTranspose1d × 4 (upsampled) → Reconstruction
```

- No fault labels required
- Anomaly threshold calibrated on healthy validation set

---

## Signal Processing Pipeline

```
Raw Signal (C, T)
       │
       ├─→ FFT Analysis       → amplitude spectrum, THD, harmonic features
       ├─→ STFT Spectrogram   → (C, H, W) time-frequency image for CNN
       ├─→ Wavelet Features   → DWT sub-band energies, CWT scalogram
       └─→ Harmonic Analysis  → IEEE 519 compliance, ITSC sideband detection
```

All transforms are accessible via the unified `SignalFeatureExtractor`:

```python
from signal_processing import SignalFeatureExtractor

extractor = SignalFeatureExtractor(f_sample=50_000, output_mode='spectrogram')
X_spectrograms = extractor.transform_batch(X_raw)   # (N, C, 128, 128)
```

---

## Datasets

### Synthetic (No Download Required)

Physically-modeled signals generated via numpy/scipy:

```python
from datasets.synthetic import InverterFaultSimulator, MotorDriveSimulator

# 3-phase inverter with IGBT fault injection
sim = InverterFaultSimulator()
signals, label = sim.generate('open_circuit_T1')   # (6, n_samples)
X, y = sim.generate_dataset(n_per_class=300, window_size=1024)

# PMSM motor drive fault simulation
motor = MotorDriveSimulator()
signals, label = motor.generate('bearing')          # (3, n_samples)
```

### Real: Kaggle Electric Motor Temperature

```bash
# Prerequisites: pip install kaggle, configure ~/.kaggle/kaggle.json
python datasets/download_scripts/setup_datasets.py --dataset motor_temp
```

**Source:** https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature

---

## Benchmark Results

Expected performance on synthetic datasets (run `benchmark_all_models.py` for exact numbers):

### Inverter Fault Detection (9 classes)

| Model | Accuracy | Macro F1 | Parameters |
|---|---|---|---|
| 1D CNN (Residual) | ~97–99% | ~0.97 | 1.2M |
| Spectrogram CNN | ~96–98% | ~0.96 | 11M |
| Transformer | ~95–97% | ~0.95 | 800K |
| BiLSTM + Attention | ~94–96% | ~0.94 | 1.8M |

### Motor Drive Fault Detection (5 classes)

| Model | Accuracy | Macro F1 | Parameters |
|---|---|---|---|
| 1D CNN (Residual) | ~98–99% | ~0.98 | 1.1M |
| Spectrogram CNN | ~97–99% | ~0.97 | 11M |
| Transformer | ~96–98% | ~0.96 | 750K |
| BiLSTM + Attention | ~95–97% | ~0.95 | 1.7M |

---

## Fault Detection Pipelines

### Switch Fault Detector (Inverter IGBT)

```python
from models import CNN1DWaveformClassifier
from fault_detection import SwitchFaultDetector

model = CNN1DWaveformClassifier(n_channels=6, n_classes=9)
# ... load trained weights ...

detector = SwitchFaultDetector(model, f_sample=100_000)
result = detector.detect(signal_window)
print(result['fault_type'], result['confidence'])

# Streaming detection
detections = detector.streaming_detect(continuous_signal)
```

### Harmonic Fault Detector (IEEE 519 Compliance)

```python
from fault_detection import HarmonicFaultDetector

detector = HarmonicFaultDetector(f_sample=50_000, voltage_class='LV')
result = detector.analyze(voltage_signal)
print(f"THD-F: {result.thd_f:.2f}%  |  Fault: {result.fault_type}")
```

### Thermal Fault Detector

```python
from fault_detection import ThermalFaultDetector

detector = ThermalFaultDetector(f_sample=50_000, autoencoder=ae_model)
detector.set_baseline(healthy_signal)
result = detector.detect(new_signal)
```

---

## Tutorials

| Notebook | Description |
|---|---|
| [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) | EDA on synthetic signals, class statistics, waveform visualization |
| [02_signal_processing.ipynb](notebooks/02_signal_processing.ipynb) | FFT, STFT, wavelet, harmonic analysis walkthrough |
| [03_model_training.ipynb](notebooks/03_model_training.ipynb) | Train and compare all 5 models, results visualization |
| [04_fault_detection_demo.ipynb](notebooks/04_fault_detection_demo.ipynb) | End-to-end fault detection with streaming inference and dashboard |

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@software{ieee_ies_ped_2026,
  author       = {{IEEE IES Industrial AI Lab}},
  title        = {{AI Power Electronics Diagnostics}},
  year         = {2026},
  url          = {https://github.com/IEEE-IES-Industrial-AI-Lab/AI-Power-Electronics-Diagnostics},
}
```

### Related References

- Trabelsi, M., et al. (2012). FPGA-based real-time power converter switch failure diagnosis. *IEEE Trans. Industrial Electronics*.
- Nandi, S., Toliyat, H. A., & Li, X. (2005). Condition monitoring and fault diagnosis of electrical motors. *IEEE Trans. Energy Conversion*.
- Nie, Y., et al. (2022). A time series is worth 64 words. *ICLR 2023*.
- Kirchgässner, W., et al. (2019). Empirical evaluation of exponentially weighted moving averages for linear thermal modeling of PMSM. *IEEE IEMDC*.
- IEEE Std 519-2022, *Recommended Practice for Harmonic Control in Electric Power Systems*.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*IEEE IES Industrial AI Lab — Advancing AI for Industrial Electronics Research*
