# Datasets

This directory contains dataset loaders, synthetic generators, and download helpers for the AI-Power-Electronics-Diagnostics framework.

---

## Synthetic Data (No Download Required)

The `synthetic/` module generates realistic power electronics signals on-the-fly using physics-informed models. No internet connection or external files are needed.

| Module | Description | Fault Classes |
|---|---|---|
| `synthetic/inverter_fault_sim.py` | 3-phase VSI with IGBT faults | healthy, open-circuit T1–T6, short-circuit, DC undervoltage |
| `synthetic/motor_drive_sim.py` | PMSM stator current simulation | healthy, phase loss, ITSC, bearing, overtemperature |
| `synthetic/fault_injector.py` | Generic fault injection for augmentation | spike, dropout, harmonic, noise burst, AM, DC drift |

**Quick start (synthetic):**
```python
from datasets.synthetic import InverterFaultSimulator, MotorDriveSimulator

# Generate a single inverter fault sample
sim = InverterFaultSimulator()
signals, label = sim.generate("open_circuit_T1")  # shape: (6, n_samples)

# Generate a full labeled dataset
X, y = sim.generate_dataset(n_per_class=200, window_size=1024)
```

---

## Real Datasets

### 1. Electric Motor Temperature (Kaggle)

**Source:** https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature  
**Reference:** Kirchgässner et al., IEEE IEMDC 2019  
**Signals:** u_q, u_d, i_q, i_d, motor_speed, torque, ambient, coolant → pm temperature  
**Task:** Binary overtemperature fault detection (or PM temperature regression)

**Download:**
```bash
# Option 1: Shell script
bash datasets/download_scripts/download_motor_temp.sh

# Option 2: Python CLI
python datasets/download_scripts/setup_datasets.py --dataset motor_temp
```

**Prerequisites:**
1. `pip install kaggle`
2. Get API credentials from https://www.kaggle.com/settings → "Create new token"
3. Save to `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`

**Usage:**
```python
from datasets.loaders import MotorTemperatureLoader

loader = MotorTemperatureLoader(
    data_dir="datasets/raw/motor_temp",
    window_size=1024,
    temp_threshold=100.0,
)
X, y = loader.load()           # fault detection (binary)
X, y = loader.load_regression()  # temperature regression
```

---

## Directory Layout

```
datasets/
├── README.md
├── __init__.py
├── loaders/
│   ├── base_loader.py          # Abstract base class
│   └── motor_temp_loader.py    # Kaggle Electric Motor Temperature
├── synthetic/
│   ├── inverter_fault_sim.py   # 3-phase inverter simulation
│   ├── motor_drive_sim.py      # PMSM current simulation
│   └── fault_injector.py       # Generic fault augmentation
├── download_scripts/
│   ├── setup_datasets.py       # One-click download CLI
│   └── download_motor_temp.sh  # Kaggle bash helper
└── raw/                        # Downloaded files go here (git-ignored)
```
