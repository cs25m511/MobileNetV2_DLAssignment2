"""Static, symmetric activation quantization with calibration and forward hooks."""
import torch
from torch import nn

class ActivationQuantizer:
    def __init__(self, bits=8, max_calibration_batches=16):
        if bits not in range(2, 9):
            raise ValueError("activation bits must be between 2 and 8")
        if max_calibration_batches < 1:
            raise ValueError("max_calibration_batches must be positive")
        self.bits = bits
        self.max_calibration_batches = max_calibration_batches
        self.scales = {}
        self.handles = []

    @staticmethod
    def _targets(model):
        # Quantize post-activation feature tensors and classifier output.
        # Other intermediate tensors remain FP32 in this reference simulation.
        return [(name, m) for name, m in model.named_modules()
                if isinstance(m, (nn.ReLU6, nn.Linear))]

    def calibrate(self, model, loader, device):
        maximums = {}
        handles = []
        for name, module in self._targets(model):
            def observer(mod, inputs, output, name=name):
                maximum = output.detach().abs().amax().item()
                maximums[name] = max(maximums.get(name, 0.0), maximum)
            handles.append(module.register_forward_hook(observer))
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                for idx, (images, _) in enumerate(loader):
                    if idx >= self.max_calibration_batches: break
                    model(images.to(device))
        finally:
            for handle in handles: handle.remove()
            model.train(was_training)
        qmax = (1 << (self.bits - 1)) - 1
        self.scales = {name:max(value / qmax, 1e-12) for name, value in maximums.items()}
        if not self.scales: raise RuntimeError("No supported activation modules found")
        return self.scales

    def attach(self, model):
        if not self.scales: raise RuntimeError("Call calibrate() or restore scales first")
        self.remove()
        qmax = (1 << (self.bits - 1)) - 1
        for name, module in self._targets(model):
            if name not in self.scales: raise ValueError(f"Missing activation scale for {name}")
            scale = self.scales[name]
            def fake_quant(mod, inputs, output, scale=scale):
                # ReLU6 may be in-place upstream, but returned tensors are new.
                return (output / scale).round().clamp(-qmax, qmax) * scale
            self.handles.append(module.register_forward_hook(fake_quant))
        return model

    def remove(self):
        for handle in self.handles: handle.remove()
        self.handles = []
