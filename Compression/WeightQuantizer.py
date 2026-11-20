"""Custom symmetric, per-tensor or per-output-channel quantization (no quantization APIs)."""
import torch

class WeightQuantizer:
    def __init__(self, bits=8, per_channel=True):
        if bits not in range(2, 9):
            raise ValueError("weight bits must be between 2 and 8")
        self.bits, self.per_channel = bits, per_channel

    def quantize(self, weights):
        w = weights.detach().float().cpu()
        per_channel = self.per_channel and w.ndim >= 2
        max_q = (1 << (self.bits - 1)) - 1
        if per_channel:
            axes = tuple(range(1, w.ndim))
            scale = w.abs().amax(dim=axes).clamp_min(1e-12) / max_q
            expanded = scale.reshape((-1,) + (1,) * (w.ndim - 1))
        else:
            scale = w.abs().max().clamp_min(1e-12).reshape(1) / max_q
            expanded = scale
        integers = torch.round(w / expanded).clamp(-max_q, max_q).to(torch.int16)
        return integers, scale, per_channel

    @staticmethod
    def dequantize(integers, scale, per_channel):
        s = scale.reshape((-1,) + (1,) * (integers.ndim - 1)) if per_channel else scale
        return integers.float() * s
