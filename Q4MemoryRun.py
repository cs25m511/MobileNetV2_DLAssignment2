"""Question 2(c), 3 and 4: memory-resident low-bit weights/activation evidence.

Run AFTER Q2Run.py, so Results/Q2_Compression/*/packed_model.pt exist.
Only existing bundles are profiled; missing bundles are reported, not invented.
The 32 explicit cases remain in Q2Run.py, the single source of truth.
"""
import csv
import json
from pathlib import Path
import torch
from torch import nn
from Compression.CompactRuntime import (load_compact_runtime, packed_model_breakdown,
                                        resident_bytes, PackedReLU6)
from Q2Run import CHECKPOINT, RESULTS, TEST_CASES, prepare, accuracy

# Edit these in source, just like Q1Run.py and Q2Run.py.
PROFILING_BATCH_SIZE = 1
VALIDATION_EVALUATION = True
WANDB_LOG = True
WANDB_PROJECT = 'Q4Run4'
MAX_VALIDATION_ACCURACY_DROP_PP = 2.0  # Optional criterion for selecting a model


def profile(model, image):
    """Profile a real forward, resident tensors and packed activation boundaries.

    Activation boundary bytes measure *actual packed uint8 objects* produced by
    PackedReLU6, not peak all-live memory. Full GPU peak includes temporary FP32
    dequantization and kernels when CUDA is available.
    """
    device = image.device
    output = {}
    model.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_alloc = torch.cuda.memory_allocated(device)
    with torch.inference_mode():
        logits = model(image)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        output['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        output['cuda_peak_forward_increment_bytes'] = max(0, output['cuda_peak_allocated_bytes'] - start_alloc)
    activations = [m for m in model.modules() if isinstance(m, PackedReLU6)]
    if activations:
        output['activation_boundary_count'] = len(activations)
        output['peak_single_packed_activation_bytes'] = max(m.last_packed_bytes for m in activations)
        output['peak_single_fp32_activation_bytes'] = max(m.last_reference_fp32_bytes for m in activations)
        output['total_packed_activation_bytes_produced'] = sum(m.last_packed_bytes for m in activations)
        output['total_fp32_activation_bytes_produced'] = sum(m.last_reference_fp32_bytes for m in activations)
        output['activation_boundary_payload_ratio'] = (
            output['total_fp32_activation_bytes_produced'] / output['total_packed_activation_bytes_produced'])
        output['peak_activation_payload_ratio'] = (
            output['peak_single_fp32_activation_bytes'] / output['peak_single_packed_activation_bytes'])
    output['resident_model_tensor_bytes'] = resident_bytes(model)
    output['inference_batch_size'] = image.shape[0]
    output['input_shape'] = list(image.shape)
    return output


def save_csv(rows, destination):
    if not rows: return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with destination.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def log_wandb_result(row):
    """Log one run per configuration for W&B's native Parallel Coordinates panel.

    Each run has the quantization choices as config and numeric results as summary.
    No chart is generated locally.
    """
    if not WANDB_LOG:
        return
    try:
        import wandb
        case_config = {
            'weight_bits': row['weight_bits'],
            'activation_bits': row['activation_bits'],
            'per_channel': row['per_channel'],
            'inference_batch_size': row['inference_batch_size'],
        }
        metrics = {
            key: float(value) if isinstance(value, float) else int(value)
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"{row['test_id']}_MEMORY",
            group='Q4_Memory_Compression',
            job_type='compression-memory-audit',
            config=case_config,
            reinit='finish_previous',
        )
        try:
            run.log(metrics)
            run.summary.update(metrics)
            run.summary['test_id'] = row['test_id']
        finally:
            run.finish()
    except Exception as exc:
        print(f"  W&B logging failed for {row['test_id']}: {exc}")


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError('Copy trained Q1 model to ' + str(CHECKPOINT))
    baseline, _, test_loader, device = prepare()
    # Existing DatasetNormalizer preserves the Q1 validation split.
    validation_loader = None
    if VALIDATION_EVALUATION:
        from dataclasses import fields
        from DataManager.ArgConfig import ArgConfig
        from DataManager.DatasetNormalizer import DatasetNormalizer
        ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        valid = {f.name for f in fields(ArgConfig)}
        cfg = ArgConfig(**{k:v for k,v in ckpt['config'].items() if k in valid})
        data = DatasetNormalizer(**{n:getattr(cfg,n) for n in (
            'data_dir','batch_size','num_workers','seed','calc_meanstd',
            'normalize_mean','normalize_std','add_crop','crop_padding',
            'horizontal_flip_prob','add_jitter','jitter_brightness','jitter_contrast',
            'jitter_saturation','jitter_hue','add_random_erasing','random_erasing_prob',
            'random_erasing_scale','random_erasing_ratio')}, download=not cfg.no_download)
        _, validation_loader, _ = data.build_loaders()
    input_image = next(iter(test_loader))[0][:PROFILING_BATCH_SIZE].to(device)
    baseline.eval()
    base_bytes = resident_bytes(baseline)
    base_profile = {'resident_model_tensor_bytes':base_bytes, 'device':str(device),
                    'input_shape':list(input_image.shape)}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.memory_allocated(device)
        with torch.inference_mode(): baseline(input_image)
        torch.cuda.synchronize(device)
        base_profile['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        base_profile['cuda_peak_forward_increment_bytes'] = max(0,base_profile['cuda_peak_allocated_bytes']-start)
    base_val = None
    if validation_loader is not None:
        base_val = accuracy(baseline,validation_loader,device)['test/top1_accuracy_percent']
        base_profile['validation_top1_percent'] = base_val
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS/'memory_baseline.json').write_text(json.dumps(base_profile,indent=2))
    old_rows = {}
    summary_path = RESULTS/'compression_summary.csv'
    if summary_path.exists():
        with summary_path.open(newline='') as f:
            old_rows = {r['test_id']:r for r in csv.DictReader(f)}
    rows = []
    missing = []
    for idx, case in enumerate(TEST_CASES,1):
        name = case['test_id']
        bundle = RESULTS/name/'packed_model.pt'
        if not bundle.exists():
            missing.append(name)
            print(f'[{idx}/{len(TEST_CASES)}] Missing packed model: {name}')
            continue
        print(f'[{idx}/{len(TEST_CASES)}] Measuring packed resident memory: {name}',flush=True)
        try:
            compact = load_compact_runtime(baseline,bundle,device)
            stats = packed_model_breakdown(compact)
            runtime = profile(compact,input_image)
            # Also verify the restored/fake-quant version on the SAME input.
            from Compression.CompressionEngine import CompressionEngine
            simulated = CompressionEngine.restore(baseline,bundle,device)
            with torch.inference_mode():
                compact_pred = compact(input_image)
                reference_pred = simulated(input_image)
            max_error = (compact_pred-reference_pred).abs().max().item()
            # In old pipeline, ReLU6 is in-place; tiny differences expected but large
            # discrepancies invalidate comparable test accuracy claims.
            if max_error > 1e-3:
                print(f'  WARNING: compact runtime differs from old inference: {max_error:.6g}')
            del simulated
            validation = (accuracy(compact,validation_loader,device)['test/top1_accuracy_percent']
                          if validation_loader is not None else None)
            test = accuracy(compact,test_loader,device)
            row = {**case, **stats, **runtime,
                   'baseline_resident_model_tensor_bytes':base_bytes,
                   'resident_model_ratio':base_bytes/stats['resident_model_tensor_bytes'],
                   'resident_model_reduction_percent':100*(1-stats['resident_model_tensor_bytes']/base_bytes),
                   'activation_scales_logical_fp32_bytes':runtime.get('activation_boundary_count',0)*4,
                   'approx_baseline_model_plus_one_activation_bytes':base_bytes + runtime.get('peak_single_fp32_activation_bytes',0),
                   'approx_compact_model_plus_one_activation_bytes':stats['resident_model_tensor_bytes'] + runtime.get('peak_single_packed_activation_bytes',0),
                   'resident_weight_plus_scales_ratio':float(old_rows.get(name,{}).get('original_conv_linear_weight_bytes',
                       sum(m.weight.numel()*m.weight.element_size() for m in baseline.modules()
                           if isinstance(m,(nn.Conv2d,nn.Linear)))))/(stats['resident_packed_weight_bytes']+stats['resident_weight_scales_bytes']),
                   'resident_weight_ratio':float(old_rows.get(name,{}).get('original_conv_linear_weight_bytes',
                       sum(m.weight.numel()*m.weight.element_size() for m in baseline.modules()
                           if isinstance(m,(nn.Conv2d,nn.Linear)))))/stats['resident_packed_weight_bytes'],
                   'validation_top1_percent':validation,
                   'baseline_validation_top1_percent':base_val,
                   'test/top1_accuracy_percent':test['test/top1_accuracy_percent'],
                   'test/loss':test['test/loss'],
                   'logit_max_abs_diff_vs_old_simulation':max_error,
                   'physical_activation_storage':'packed ReLU6 boundary tensors; transient FP32 decode',
                   'full_inference_peak_measurement':'CUDA max allocated' if device.type=='cuda' else 'not measured on CPU'}
            row['approx_model_plus_activation_ratio'] = (
                row['approx_baseline_model_plus_one_activation_bytes'] /
                row['approx_compact_model_plus_one_activation_bytes'])
            row['approx_compact_model_plus_one_activation_mb'] = (
                row['approx_compact_model_plus_one_activation_bytes'] / 1e6)
            if validation is not None:
                row['validation_accuracy_drop_pp']=base_val-validation
            if 'cuda_peak_forward_increment_bytes' in runtime and 'cuda_peak_forward_increment_bytes' in base_profile:
                b=base_profile['cuda_peak_forward_increment_bytes']; c=runtime['cuda_peak_forward_increment_bytes']
                row['measured_cuda_forward_peak_ratio']=b/c if c else None
            rows.append(row)
            (RESULTS/name/'memory_profile.json').write_text(json.dumps(row,indent=2))
            print(f"  resident model: {stats['resident_model_tensor_bytes']/1e6:.3f} MB"
                  f" | test {test['test/top1_accuracy_percent']:.2f}% | diff {max_error:.4g}")
            save_csv(rows, RESULTS/'memory_compression_summary.csv')
            log_wandb_result(row)
        except Exception as exc:
            missing.append(name + ': '+repr(exc))
            print('  FAILED:',repr(exc),flush=True)
    (RESULTS/'memory_profile_failures.json').write_text(json.dumps(missing,indent=2))
    if not rows:
        raise RuntimeError('No models profiled. Copy Q2Run.py packed_model.pt outputs to Results/Q2_Compression.')
    # Selection is based on validation, not test; size threshold can be refined.
    if base_val is not None:
        acceptable = [r for r in rows if r['validation_accuracy_drop_pp'] <= MAX_VALIDATION_ACCURACY_DROP_PP]
        if acceptable:
            best = min(acceptable, key=lambda r:r['resident_model_tensor_bytes'])
            (RESULTS/'validation_selected_memory_model.json').write_text(json.dumps(best,indent=2))
    # W&B Parallel Coordinates is created in the W&B UI from the per-case runs.
    print('Saved:',RESULTS/'memory_compression_summary.csv')
    print('Profiled:',len(rows),'/',len(TEST_CASES))
    print('NOTE: activation ratios describe packed ReLU6 boundary payloads; NOT complete peak memory.')


if __name__ == '__main__':
    main()