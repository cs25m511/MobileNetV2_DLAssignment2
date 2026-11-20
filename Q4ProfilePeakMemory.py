"""Fresh-process inference GPU memory comparison for Q2/Q4.

Examples:
  python Q4ProfilePeakMemory.py --cases Q2TC03_W4_A8 Q2TC01_W8_A8
  python Q4ProfilePeakMemory.py --all

Runs baseline and each compressed case in a separate Python process to avoid
co-resident baseline weights and allocator/workspace contamination. Measures
PyTorch CUDA allocated bytes; CUDA drivers and other processes aren't counted.
"""
import argparse
import csv
import gc
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / 'Results' / 'Q2_Compression'


def _measure_case(name, warmup=5, repeat=10):
    import torch
    from Q2Run import prepare, TEST_CASES, RESULTS
    from Compression.CompactRuntime import load_compact_runtime, resident_bytes

    baseline, _, test_loader, device = prepare()
    if device.type != 'cuda':
        raise RuntimeError('CUDA is required for this GPU memory measurement; CPU fallback would not measure VRAM.')
    image = next(iter(test_loader))[0][:1].to(device)
    if name == 'FP32':
        model = baseline
    else:
        case_ids = {c['test_id'] for c in TEST_CASES}
        if name not in case_ids:
            raise ValueError(f'Unknown case {name}')
        bundle = RESULTS / name / 'packed_model.pt'
        if not bundle.exists():
            raise FileNotFoundError(str(bundle))
        model = load_compact_runtime(baseline, bundle, device)
        baseline.cpu()
        del baseline
        gc.collect()
        torch.cuda.empty_cache()

    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(image)
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
        # Keep model and input resident while measuring the steady-state forward.
        # Excludes temporary allocations from loading/converting the model.
        start = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repeat):
            output = model(image)
            del output
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        end = torch.cuda.memory_allocated(device)
    return {
        'case':name, 'device':str(device), 'batch_size':1,
        'model_resident_tensor_bytes':resident_bytes(model),
        'starting_allocated_bytes':start,
        'peak_allocated_bytes':peak,
        'incremental_forward_peak_bytes':max(0,peak-start),
        'end_allocated_bytes':end,
        'warmup_passes':warmup,'measured_passes':repeat,
        'note':'PyTorch GPU allocation (not entire GPU/driver usage)'
    }


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',default=None,help=argparse.SUPPRESS)
    parser.add_argument('--cases',nargs='*',default=[],help='Compression test IDs to profile')
    parser.add_argument('--all',action='store_true',help='Profile all Q2Run.TEST_CASES')
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--repeat',type=int,default=10)
    args=parser.parse_args()
    if args.worker is not None:
        try:
            result=_measure_case(args.worker,args.warmup,args.repeat)
            print('PEAK_PROFILE_JSON='+json.dumps(result),flush=True)
        except Exception as exc:
            print('PEAK_PROFILE_ERROR='+repr(exc),flush=True)
            raise
        return
    from Q2Run import TEST_CASES
    cases=([c['test_id'] for c in TEST_CASES] if args.all else args.cases)
    if not cases:
        parser.error('Provide --cases <ID> [ID ...] or --all')
    records=[]
    RESULT_DIR.mkdir(parents=True,exist_ok=True)
    for name in ['FP32',*cases]:
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',name,
             '--warmup',str(args.warmup),'--repeat',str(args.repeat)]
        completed=subprocess.run(cmd,cwd=str(ROOT),text=True,capture_output=True)
        line=next((x for x in completed.stdout.splitlines() if x.startswith('PEAK_PROFILE_JSON=')),None)
        if completed.returncode or line is None:
            print('Failed',name,completed.stderr[-1200:] or completed.stdout[-1200:])
            continue
        record=json.loads(line.split('=',1)[1]); records.append(record)
        print(f"{name}: resident={record['model_resident_tensor_bytes']/1e6:.2f}MB; "
              f"peak live={record['peak_allocated_bytes']/1e6:.2f}MB; "
              f"forward increment={record['incremental_forward_peak_bytes']/1e6:.2f}MB")
    if not records:
        raise RuntimeError('No results; check CUDA and saved bundles')
    fp32=next((r for r in records if r['case']=='FP32'),None)
    if fp32:
        for row in records:
            row['peak_total_ratio_vs_fp32'] = (fp32['peak_allocated_bytes']/row['peak_allocated_bytes']
                                                  if row['peak_allocated_bytes'] else None)
            row['resident_ratio_vs_fp32'] = (fp32['model_resident_tensor_bytes']/row['model_resident_tensor_bytes']
                                              if row['model_resident_tensor_bytes'] else None)
    output=RESULT_DIR/'isolated_gpu_peak_memory.csv'
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in records for k in row)))
        writer.writeheader();writer.writerows(records)
    print('Saved:',output)


if __name__=='__main__':
    run()
