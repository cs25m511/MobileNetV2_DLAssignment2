# CS6886W Assignment 2 — MobileNetV2 Compression on CIFAR-10
**Submitted By** Jitender Kumar (CS25M511).

**Course:** Systems Engineering for Deep Learning (CS6886W), IIT Madras  
**Task:** Train MobileNetV2 on CIFAR-10, apply manually implemented post-training weight/activation quantization, and assess accuracy and **in-memory** compression.

> **Repository URL:** Add the actual public GitHub URL here before submission. No repository URL has been verified in this document.

## 1. Overview

The project uses a custom MobileNetV2 implementation and manual post-training quantization (PTQ). The compressed runtime stores weights in packed `uint8` buffers, packs selected ReLU6 activation outputs, and unpacks/dequantizes them when floating-point operations are needed. Built-in quantization/compression APIs are **not** used.

- Baseline selected by validation accuracy: `TC04_dropout=0.0`.
- Weight and activation bit-widths: **2, 4, 6, 8**.
- Weight scaling: **per-channel (PC)** or **per-tensor (PT)**.
- Compression sweep: **32 configurations** (`4 × 4 × 2`).
- Selected accuracy–compression trade-off: `Q2TC03_W4_A8` (**W4A8 PC**).
- Memory assessment: actual resident tensor payloads and independently measured peak PyTorch CUDA allocation. Disk bundle size is recorded separately and **is not** substituted for RAM/VRAM usage.

## 2. Project layout

```text
.
├── Q1Run.py                       # 12 baseline training configurations
├── Q2Run.py                       # 32 PTQ configurations and test evaluation
├── Q2LayerReport.py               # Quantized / excluded layer inventory
├── Q4MemoryRun.py                 # Resident memory, activation payloads, accuracy
├── Q4ProfilePeakMemory.py           # FP32 vs. packed runtime in fresh CUDA processes
├── ExportEnvironment.py           # Record environment and checkpoint config
├── requirements.txt
├── CIFAR10Dataset                 # CIFAR10 Dataset
├── Control/
│   └── Control.py
├── DataManager/
│   ├── ArgConfig.py
│   └── DatasetNormalizer.py
├── Models/
│   ├── MobileNetV2.py
│   ├── Optimizers.py
│   └── Trainer.py
├── Compression/
│   ├── WeightQuantizer.py
│   ├── ActivationQuantizer.py
│   ├── BitPacker.py
│   ├── CompressionEngine.py
│   └── CompactRuntime.py
├── WandbLogger/
│   └── WandbLogger.py
└── Results/                       # Generated on execution / copied from experiments
    ├── TC04_dropout=0.0/
    │   └── best_model.pt
    └── Q2_Compression/
        ├── compression_summary.csv
        ├── memory_compression_summary.csv
        ├── isolated_gpu_peak_memory.csv
        └── Q2TC03_W4_A8/
            └── packed_model.pt
```

Some files are optional depending on which project revision is checked out. The commands below assume the filenames above from the supplied project archive.

## 3. Environment and dependencies

### Requirements

- Python **3.10 or newer** recommended (use the exact version recorded in the experiment environment for full reproduction).
- PyTorch with a CUDA-enabled build and a compatible NVIDIA driver **for VRAM profiling**. CPU operation is sufficient for most training/testing scripts but **not** the CUDA peak-memory benchmark.
- CIFAR-10 (the loader can download it when `no_download=False`).

The supplied `requirements.txt` currently contains **minimum** versions, not a pinned environment:

```text
torch>=2.2
torchvision>=0.17
numpy>=1.24
wandb>=0.17
matplotlib>=3.8
```

Set up the environment on Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For exact reproducibility, export the package versions **in the environment that produced the submitted results**:

```bash
python -m pip freeze > environment_versions.txt
python ExportEnvironment.py
```

The environment report is written to `Results/Q2_Compression/environment_reproducibility.json`; it includes Python/framework versions, hardware information, and the selected checkpoint configuration when the checkpoint exists. Commit `environment_versions.txt` and the JSON report alongside the submission. Do not substitute the versions from another machine.

To enable online W&B logging, run `wandb login` and verify the relevant `WANDB_MODE`/`wandb_mode` settings. Q2 compression runs use W&B project **`DLAssignment2`**; the Q1 source defines **`CS6886W_Assignment2_MobileNetV2`**. These can be changed in source if needed.

## 4. Reproducibility configuration

| Setting | Value / source |
|---|---|
| Main random seed | `42` in `Q1Run.py` / `ArgConfig` |
| Model | Custom MobileNetV2, CIFAR-10, 10 classes |
| Selected baseline | `Results/TC04_dropout=0.0/best_model.pt` |
| Data split | 45,000 training / 5,000 validation / 10,000 test |
| Dataset split reproducibility | Seeded `torch.Generator().manual_seed(cfg.seed)` |
| Training epochs | 100 |
| Training batch size | 128 for selected TC04 |
| Optimizer | SGD, LR 0.05, momentum 0.9, weight decay 0.0005 |
| Learning-rate schedule | Cosine |
| Width multiplier | 1.0 |
| Dropout for TC04 | 0.0 |
| Batch normalization | Enabled |
| Preprocessing | ToTensor, calculated/cached per-channel normalization |
| Training augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip(0.5), ColorJitter as configured |
| Activation calibration | 16 non-shuffled training-subset batches (`CALIBRATION_BATCHES=16`) |
| Compression widths | Weight and activation bits in `{2,4,6,8}` |
| Weight scaling | Per-channel or per-tensor |
| GPU profiling input | One image of shape `(1,3,32,32)` |
| GPU profiling repetitions | 10 warm-up; 20 measured passes in the commands below |

`Control/Control.py` seeds PyTorch and CUDA and sets `torch.backends.cudnn.deterministic=True` and `torch.backends.cudnn.benchmark=False` during training. Exact numerical repeatability may still depend on installed versions, hardware, and CUDA operations. For the compression sweep, `Q2Run.py` reloads the checkpoint's configuration and selects calibration examples from training data only.

**Important:** `Q2Run.py` has `SKIP_COMPLETED=True`. A rerun can reuse existing `result.json` and `packed_model.pt`; to demonstrate *independent* repeatability, use a clean `Results/Q2_Compression/` or turn off skipping and preserve the previous outputs separately. Record checkpoint and bundle hashes when comparing different runs.

## 5. Step-by-step reproduction

Run these commands from the project root.

### Step 1 — Train baseline and choose the validation winner (Q1)

```bash
python Q1Run.py
```

`Q1Run.py` evaluates twelve controlled training cases (potentially time-consuming). The selected model for the compression experiment is expected at:

```text
Results/TC04_dropout=0.0/best_model.pt
```

If using the supplied trained checkpoint instead, put it at that exact path. It must match the MobileNetV2 architecture and checkpoint structure expected by `Q2Run.py` (`model_state_dict`, `config`).

### Step 2 — Run manual compression and test evaluation (Q2 / Q3)

```bash
python Q2Run.py
```

This runs 32 named configurations. For each case, it saves `Results/Q2_Compression/<test_id>/packed_model.pt` and `result.json` and builds:

```text
Results/Q2_Compression/compression_summary.csv
```

The summary contains bit-widths, scaling method, packed-weight bytes, auxiliary/scale estimates, serialized-bundle bytes, test accuracy and the FP32 baseline test accuracy. **Q2Run's reference evaluation is fake-quantized floating-point inference; the separate compact runtime is used for in-memory payload measurement.**

### Step 3 — Inspect compression layer coverage

```bash
python Q2LayerReport.py
```

Output: `Results/Q2_Compression/layer_compression_scope.csv`. The inspected selected architecture has 52 Conv2d and one Linear layer with quantized weights, 35 ReLU6 activation boundaries, and retained FP32 operations/metadata such as BatchNorm.

### Step 4 — Measure in-memory model and activation payloads (Q4)

```bash
python Q4MemoryRun.py
```

Output: `Results/Q2_Compression/memory_compression_summary.csv` and per-case `memory_profile.json` files. The script evaluates packed-runtime resident tensor storage, scaling overhead, and the bytes of packed ReLU6 activation boundaries, along with CIFAR-10 accuracy. Set the W&B options at the top of the script as needed.

**Measurement limits:** Resident tensor payloads exclude Python-object and allocator overhead. ReLU6 compression ratios apply to packed **boundaries**, not to total live activation memory. The script's in-process CUDA peak data may include other resident models; use the isolated profiler below for the final peak-memory comparison.

### Step 5 — Compare original FP32 and selected compressed model in isolated processes

```bash
python ProfilePeakMemory.py \
  --cases Q2TC03_W4_A8 \
  --warmup 10 \
  --repeat 20
```

Output:

```text
Results/Q2_Compression/isolated_gpu_peak_memory.csv
```

The script automatically includes the **original unquantized FP32 model** and separately profiles `Q2TC03_W4_A8` using fresh Python processes. The important metric is `peak_allocated_bytes` (peak **PyTorch** CUDA allocation for warmed-up inference); `incremental_forward_peak_bytes` measures only the additional allocation above the starting point.

**CUDA is required.** Measurements exclude CUDA driver allocations, other processes and model-load peaks.

## 6. W&B Parallel Coordinates chart (Q3)

The chart is generated **in W&B**, not using a locally rendered Matplotlib chart. Open the runs from `DLAssignment2`, select the completed Q2/Q4 experiments and create a **Parallel Coordinates** panel. Include axes for:

- Weight precision (`weight_bits`)
- Activation precision (`activation_bits`)
- Per-channel/per-tensor scaling (`per_channel`)
- Validation top-1 accuracy (where logged)
- Test top-1 accuracy (`test/top1_accuracy_percent`)
- Resident model compression (`resident_model_ratio`)
- Effective weight compression (`resident_weight_plus_scales_ratio`)
- Packed activation-boundary compression (`activation_boundary_payload_ratio`)

If the axes do not appear, inspect each run's *Config* and *History/Summary* and ensure that the required fields were logged in the same project and comparison group. Include an exported screenshot of the final W&B panel in the PDF submission.

## 7. Recorded results from supplied experiment CSVs

These are **recorded experimental values**, not values computed by running the scripts while preparing this README.

| Metric | FP32 baseline | W4A8 per-channel |
|---|---:|---:|
| CIFAR-10 test accuracy (consistent Q2 summaries) | 91.59% | 90.10% |
| Resident model tensor payload | 9,083,592 B | 1,442,896 B |
| Effective weight compression including weight scales | 1.00× | 7.53× |
| Packed ReLU6 activation-boundary compression | 1.00× | 4.00× |
| Resident model tensor compression | 1.00× | 6.30× |
| Peak PyTorch CUDA allocation (isolated) | 18,548,736 B | 12,126,208 B |
| Peak allocation ratio | 1.00× | 1.53× |

The peak allocated-memory reduction is about **34.63%** (computed from raw bytes); minor last-decimal differences in separately rounded summaries should not be interpreted as experimental differences. The resident memory reduction is approximately **84.12%**. The FP32 and packed models must be compared under identical input and profiler settings.

`compression_summary.csv` gives Q2 fake-quantization evaluation results, whereas `memory_compression_summary.csv` gives compact-runtime results. Do not silently mix accuracy values from independently generated bundles or calibration sweeps; for a final claim, link all outputs to the specific compressed bundle and original checkpoint (ideally record their SHA-256 hashes).
