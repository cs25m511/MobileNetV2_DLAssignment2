"""Q2(a): 32 explicitly named compression experiments (supported bit-widths only).

Run: python Q2Run.py
Edit TEST_CASES below to select or modify experiments.
"""
import csv
import json
from dataclasses import fields
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from DataManager.ArgConfig import ArgConfig
from DataManager.DatasetNormalizer import DatasetNormalizer
from Models.MobileNetV2 import MobileNetV2
from Compression.CompressionEngine import CompressionEngine
from WandbLogger.WandbLogger import WandbLogger

# ----------------------- 1. EXPERIMENT SETTINGS -----------------------
CHECKPOINT = Path('Results/TC04_dropout=0.0/best_model.pt')
RESULTS = Path('Results/Q2_Compression')
WANDB_PROJECT = 'DLA2Q2MemRun4'
WANDB_MODE = 'online'               # online | offline | disabled
CALIBRATION_BATCHES = 16
SKIP_COMPLETED = True
# Named, explicit test cases (same style as Q1Run.py).
# PC = per-channel weights, PT = per-tensor weights.
# Supported weight and activation bit widths: 2, 4, 6, 8.
TEST_CASES = [
    {"test_id": "Q2TC01_W8_A8", "weight_bits": 8, "activation_bits": 8, "per_channel": True},
    {"test_id": "Q2TC02_W6_A8", "weight_bits": 6, "activation_bits": 8, "per_channel": True},
    {"test_id": "Q2TC03_W4_A8", "weight_bits": 4, "activation_bits": 8, "per_channel": True},
    {"test_id": "Q2TC04_W8_A6", "weight_bits": 8, "activation_bits": 6, "per_channel": True},
    {"test_id": "Q2TC05_W8_A4", "weight_bits": 8, "activation_bits": 4, "per_channel": True},
    {"test_id": "Q2TC06_W4_A4", "weight_bits": 4, "activation_bits": 4, "per_channel": True},
    {"test_id": "Q2TC07_W2_A2_PC", "weight_bits": 2, "activation_bits": 2, "per_channel": True},
    {"test_id": "Q2TC08_W2_A2_PT", "weight_bits": 2, "activation_bits": 2, "per_channel": False},
    {"test_id": "Q2TC09_W2_A4_PC", "weight_bits": 2, "activation_bits": 4, "per_channel": True},
    {"test_id": "Q2TC10_W2_A4_PT", "weight_bits": 2, "activation_bits": 4, "per_channel": False},
    {"test_id": "Q2TC11_W2_A6_PC", "weight_bits": 2, "activation_bits": 6, "per_channel": True},
    {"test_id": "Q2TC12_W2_A6_PT", "weight_bits": 2, "activation_bits": 6, "per_channel": False},
    {"test_id": "Q2TC13_W2_A8_PC", "weight_bits": 2, "activation_bits": 8, "per_channel": True},
    {"test_id": "Q2TC14_W2_A8_PT", "weight_bits": 2, "activation_bits": 8, "per_channel": False},
    {"test_id": "Q2TC15_W4_A2_PC", "weight_bits": 4, "activation_bits": 2, "per_channel": True},
    {"test_id": "Q2TC16_W4_A2_PT", "weight_bits": 4, "activation_bits": 2, "per_channel": False},
    {"test_id": "Q2TC17_W4_A4_PT", "weight_bits": 4, "activation_bits": 4, "per_channel": False},
    {"test_id": "Q2TC18_W4_A6_PC", "weight_bits": 4, "activation_bits": 6, "per_channel": True},
    {"test_id": "Q2TC19_W4_A6_PT", "weight_bits": 4, "activation_bits": 6, "per_channel": False},
    {"test_id": "Q2TC20_W4_A8_PT", "weight_bits": 4, "activation_bits": 8, "per_channel": False},
    {"test_id": "Q2TC21_W6_A2_PC", "weight_bits": 6, "activation_bits": 2, "per_channel": True},
    {"test_id": "Q2TC22_W6_A2_PT", "weight_bits": 6, "activation_bits": 2, "per_channel": False},
    {"test_id": "Q2TC23_W6_A4_PC", "weight_bits": 6, "activation_bits": 4, "per_channel": True},
    {"test_id": "Q2TC24_W6_A4_PT", "weight_bits": 6, "activation_bits": 4, "per_channel": False},
    {"test_id": "Q2TC25_W6_A6_PC", "weight_bits": 6, "activation_bits": 6, "per_channel": True},
    {"test_id": "Q2TC26_W6_A6_PT", "weight_bits": 6, "activation_bits": 6, "per_channel": False},
    {"test_id": "Q2TC27_W6_A8_PT", "weight_bits": 6, "activation_bits": 8, "per_channel": False},
    {"test_id": "Q2TC28_W8_A2_PC", "weight_bits": 8, "activation_bits": 2, "per_channel": True},
    {"test_id": "Q2TC29_W8_A2_PT", "weight_bits": 8, "activation_bits": 2, "per_channel": False},
    {"test_id": "Q2TC30_W8_A4_PT", "weight_bits": 8, "activation_bits": 4, "per_channel": False},
    {"test_id": "Q2TC31_W8_A6_PT", "weight_bits": 8, "activation_bits": 6, "per_channel": False},
    {"test_id": "Q2TC32_W8_A8_PT", "weight_bits": 8, "activation_bits": 8, "per_channel": False},
]

# Uncomment this line to run only a few cases while testing:
# TEST_CASES = TEST_CASES[:6]


def accuracy(model, loader, device):
    """Calculate mean cross-entropy and top-1 accuracy."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss_sum += nn.functional.cross_entropy(logits, targets, reduction='sum').item()
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.numel()
    return {'test/loss': loss_sum / total,
            'test/accuracy': correct / total,
            'test/top1_accuracy_percent': 100 * correct / total,
            'test/correct': correct, 'test/total': total}


def prepare():
    """Load Q1 checkpoint, original data split, and training-only calibration."""
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    valid = {f.name for f in fields(ArgConfig)}
    cfg = ArgConfig(**{k: v for k, v in ckpt['config'].items() if k in valid})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = DatasetNormalizer(
        **{name: getattr(cfg, name) for name in (
            'data_dir', 'batch_size', 'num_workers', 'seed', 'calc_meanstd',
            'normalize_mean', 'normalize_std', 'add_crop', 'crop_padding',
            'horizontal_flip_prob', 'add_jitter', 'jitter_brightness',
            'jitter_contrast', 'jitter_saturation', 'jitter_hue',
            'add_random_erasing', 'random_erasing_prob',
            'random_erasing_scale', 'random_erasing_ratio')},
        download=not cfg.no_download)
    _, _, test_loader = data.build_loaders()

    baseline = MobileNetV2(num_classes=10, width_mult=cfg.width_mult,
                           dropout=cfg.dropout, batch_norm=cfg.use_batchnorm).to(device)
    baseline.load_state_dict(ckpt['model_state_dict'])
    baseline.eval()

    # Reconstruct the seeded 45k training indices; never calibrate on val/test.
    indices = torch.randperm(50000, generator=torch.Generator().manual_seed(cfg.seed)).tolist()[5000:]
    mean, std = (data._get_mean_std() if cfg.calc_meanstd
                 else (cfg.normalize_mean, cfg.normalize_std))
    raw = datasets.CIFAR10(root=cfg.data_dir, train=True, download=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(), transforms.Normalize(mean, std)]))
    selected = Subset(raw, indices[:CALIBRATION_BATCHES * cfg.batch_size])
    calibration = DataLoader(selected, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    return baseline, calibration, test_loader, device


def save_csv(rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (RESULTS / 'compression_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f'Q1 checkpoint not found: {CHECKPOINT}')
    baseline, calib, test, device = prepare()
    baseline_metrics = accuracy(baseline, test, device)
    (RESULTS / 'baseline_test_metrics.json').write_text(json.dumps(baseline_metrics, indent=2))

    cases, rows, failures = TEST_CASES, [], []
    allowed = {2, 4, 6, 8}
    if any(c["weight_bits"] not in allowed or c["activation_bits"] not in allowed for c in cases):
        raise ValueError("Only bit widths 2, 4, 6, 8 are supported")
    if len({(c["weight_bits"], c["activation_bits"], c["per_channel"]) for c in cases}) != len(cases):
        raise ValueError("Duplicate compression test cases")
    print(f'Running {len(cases)} supported compression test cases (W/A: 2,4,6,8)')
    for i, case in enumerate(cases, start=1):
        name = case['test_id']
        folder = RESULTS / name
        folder.mkdir(exist_ok=True)
        report = folder / 'result.json'
        print(f'[{i}/{len(cases)}] {name}')

        if SKIP_COMPLETED and report.exists() and (folder / "packed_model.pt").exists():
            try:
                saved = json.loads(report.read_text())
                if all(saved.get(k) == v for k, v in case.items()):
                    rows.append(saved)
                    save_csv(rows)
                    print('  Already completed; skipped.')
                    continue
            except (OSError, ValueError):
                pass

        logger = None
        try:
            logger = WandbLogger(WANDB_MODE != 'disabled', WANDB_PROJECT,
                                 config=case, name=name, mode=WANDB_MODE,
                                 group='Q2_Compression', job_type='compression')
            engine = CompressionEngine(case['weight_bits'], case['activation_bits'],
                                       case['per_channel'], CALIBRATION_BATCHES)
            compressed, sizes = engine.compress(baseline, calib, device,
                                                folder / 'packed_model.pt')
            metrics = accuracy(compressed, test, device)
            row = {**case, **sizes, **metrics,
                   'baseline_test_accuracy_percent': baseline_metrics['test/top1_accuracy_percent'],
                   'accuracy_drop_pp': baseline_metrics['test/top1_accuracy_percent']
                                       - metrics['test/top1_accuracy_percent']}
            report.write_text(json.dumps(row, indent=2))
            rows.append(row)
            numbers = {k: v for k, v in row.items() if isinstance(v, (int, float))}
            logger.log_epoch(numbers)
            logger.log_summary(numbers)
            print(f"  Top-1: {metrics['test/top1_accuracy_percent']:.2f}%")
        except Exception as e:
            failures.append({'test_id': name, 'error': str(e)})
            (folder / 'error.json').write_text(json.dumps(failures[-1], indent=2))
            print(f'  FAILED: {e}')
        finally:
            if logger is not None:
                logger.finish()
            save_csv(rows)

    (RESULTS / 'failed_cases.json').write_text(json.dumps(failures, indent=2))
    print(f'Done: {len(rows)}/{len(cases)} results. Failed this session: {len(failures)}')
    print('Summary:', RESULTS / 'compression_summary.csv')


if __name__ == '__main__':
    main()