"""Reference post-training weight+activation quantization and real weight bit-storage.

Evaluation uses float PyTorch kernels with quantize/dequantize simulation. Thus
model weight payload is genuinely packed, but runtime activations are NOT stored
as packed integers and CPU/GPU latency is not claimed to improve.
"""
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from .WeightQuantizer import WeightQuantizer
from .ActivationQuantizer import ActivationQuantizer
from .BitPacker import BitPacker

class CompressionEngine:
    def __init__(self, weight_bits=8, activation_bits=8, per_channel=True, calibration_batches=16):
        self.wq = WeightQuantizer(weight_bits, per_channel)
        self.aq = ActivationQuantizer(activation_bits, calibration_batches)
        self.packed = {}

    def compress(self, baseline, calibration_loader, device, filepath):
        # Deep-copy: never mutate the original uncompressed model.
        model = deepcopy(baseline).to(device).eval()
        self.aq.calibrate(model, calibration_loader, device)
        entries = {}
        # Only actual Conv2d/Linear weights are compressed; BN and bias stay FP32.
        for module_name, module in model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)): continue
            ints, scale, per_channel = self.wq.quantize(module.weight)
            name = module_name + '.weight'
            entries[name] = {
                'shape': tuple(ints.shape), 'count': ints.numel(),
                'bits': self.wq.bits, 'scale': scale.clone(),
                'per_channel': per_channel, 'packed': BitPacker.pack(ints, self.wq.bits)
            }
            with torch.no_grad():
                module.weight.copy_(self.wq.dequantize(ints, scale, per_channel).to(device))
        self.packed = entries
        # Float32 BN parameters / biases / running stats are retained for exact reconstruction.
        other = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                 if key not in entries}
        bundle = {
            'format': 'manual-ptq-v1', 'weight_bits': self.wq.bits,
            'activation_bits': self.aq.bits, 'per_channel': self.wq.per_channel,
            'weights': entries, 'other_state': other, 'activation_scales': self.aq.scales
        }
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, filepath)
        self.aq.attach(model)
        original_bytes = sum(t.numel() * t.element_size() for t in baseline.state_dict().values())
        packed_weight_bytes = sum(len(e['packed']) for e in entries.values())
        uncompressed_weight_bytes = sum(e['count'] * 4 for e in entries.values())
        auxiliary_bytes = sum(t.numel() * t.element_size() for t in other.values())
        scales_bytes = sum(e['scale'].numel() * 4 for e in entries.values()) + len(self.aq.scales) * 4
        return model, {
            'weight_bits':self.wq.bits, 'activation_bits':self.aq.bits,
            'original_state_bytes':original_bytes,
            'original_conv_linear_weight_bytes':uncompressed_weight_bytes,
            'packed_conv_linear_weight_bytes':packed_weight_bytes,
            'auxiliary_fp32_bytes':auxiliary_bytes,
            'scales_approx_bytes':scales_bytes,
            'estimated_compact_bytes':packed_weight_bytes + auxiliary_bytes + scales_bytes,
            'actual_bundle_bytes':filepath.stat().st_size,
            'model_file':str(filepath)
        }

    @staticmethod
    def restore(baseline_architecture, filepath, device):
        bundle = torch.load(filepath, map_location='cpu', weights_only=True)
        if bundle['format'] != 'manual-ptq-v1': raise ValueError("Unexpected compression format")
        state = dict(bundle['other_state'])
        for name, e in bundle['weights'].items():
            ints = BitPacker.unpack(e['packed'], e['count'], e['bits'], e['shape'])
            state[name] = WeightQuantizer.dequantize(ints, e['scale'], e['per_channel'])
        model = deepcopy(baseline_architecture).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        aq = ActivationQuantizer(bundle['activation_bits'])
        aq.scales = bundle['activation_scales']
        aq.attach(model)
        # Keep quantizer live to keep activation hooks attached.
        model._activation_quantizer = aq
        return model
