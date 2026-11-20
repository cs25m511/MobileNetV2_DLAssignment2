"""Low-bit *resident* weights and ReLU6 activations for the custom MobileNetV2.

No quantization/compression libraries. Float kernels unpack weights/inputs on demand.
This demonstrates packed retained tensors, NOT integer kernels or lower peak GPU memory.
"""
from copy import deepcopy
from types import MethodType
import math
import torch
from torch import nn
from torch.nn import functional as F


def packed_bytes(count, bits):
    return (int(count) * int(bits) + 7) // 8


def pack_signed(q, bits):
    """Pack 2/4/6/8-bit two's-complement codes, MSB first.

    A byte-group implementation: no O(num_values * bits) int64 index tensor.
    Matches the existing manual-ptq-v1 file encoding.
    """
    if bits not in (2, 4, 6, 8):
        raise ValueError('bits must be 2, 4, 6, or 8')
    codes = q.reshape(-1).to(torch.uint8) & ((1 << bits) - 1)
    count = codes.numel()
    if not count:
        return codes
    if bits == 8:
        return codes.contiguous()
    values_per_group = {2: 4, 4: 2, 6: 4}[bits]
    pad = (-count) % values_per_group
    if pad:
        codes = torch.cat((codes, codes.new_zeros(pad)))
    if bits == 4:
        return ((codes[0::2] << 4) | codes[1::2]).contiguous()
    if bits == 2:
        return ((codes[0::4] << 6) | (codes[1::4] << 4) |
                (codes[2::4] << 2) | codes[3::4]).contiguous()
    # Four 6-bit values occupy three bytes.
    b0, b1, b2, b3 = codes[0::4], codes[1::4], codes[2::4], codes[3::4]
    packed = torch.empty((codes.numel() // 4) * 3, dtype=torch.uint8, device=codes.device)
    packed[0::3] = (b0 << 2) | (b1 >> 4)
    packed[1::3] = (b1 << 4) | (b2 >> 2)
    packed[2::3] = (b2 << 6) | b3
    return packed[:packed_bytes(count, bits)].contiguous()


def unpack_signed(payload, count, bits, shape):
    """Unpack packed codes using byte/nibble operations, without bit indices."""
    if bits not in (2, 4, 6, 8):
        raise ValueError('bits must be 2, 4, 6, or 8')
    if math.prod(shape) != count:
        raise ValueError('shape does not match count')
    if payload.numel() < packed_bytes(count, bits):
        raise ValueError('Truncated packed tensor')
    if count == 0:
        return torch.empty(shape, dtype=torch.int8, device=payload.device)
    if bits == 8:
        codes = payload[:count]
    elif bits == 4:
        n = (count + 1) // 2
        data = payload[:n]
        codes = torch.empty(n * 2, dtype=torch.uint8, device=payload.device)
        codes[0::2] = data >> 4
        codes[1::2] = data & 15
    elif bits == 2:
        n = (count + 3) // 4
        data = payload[:n]
        codes = torch.empty(n * 4, dtype=torch.uint8, device=payload.device)
        codes[0::4] = data >> 6
        codes[1::4] = (data >> 4) & 3
        codes[2::4] = (data >> 2) & 3
        codes[3::4] = data & 3
    else:  # 6-bit: four codes per three bytes
        n = (count + 3) // 4
        data = payload[:packed_bytes(count, 6)]
        if data.numel() < n * 3:
            data = torch.cat((data, data.new_zeros(n * 3 - data.numel())))
        b0, b1, b2 = data[0::3], data[1::3], data[2::3]
        codes = torch.empty(n * 4, dtype=torch.uint8, device=payload.device)
        codes[0::4] = b0 >> 2
        codes[1::4] = ((b0 & 3) << 4) | (b1 >> 4)
        codes[2::4] = ((b1 & 15) << 2) | (b2 >> 6)
        codes[3::4] = b2 & 63
    codes = codes[:count]
    # Sign-extend using 8-bit arithmetic (no large int32/int64 intermediates).
    return ((codes << (8 - bits)).to(torch.int8) >> (8 - bits)).reshape(shape)


class PackedActivation:
    """An actual packed uint8 buffer; FP32 values are not retained."""
    def __init__(self, data, shape, bits, scale):
        self.data, self.shape, self.bits, self.scale = data, tuple(shape), int(bits), float(scale)
    @property
    def nbytes(self):
        return self.data.numel() * self.data.element_size()
    @property
    def reference_fp32_bytes(self):
        return math.prod(self.shape) * 4
    def decode(self):
        return unpack_signed(self.data, math.prod(self.shape), self.bits, self.shape).float() * self.scale


def decode(x):
    return x.decode() if isinstance(x, PackedActivation) else x


class PackedWeightBase(nn.Module):
    def __init__(self, source, entry):
        super().__init__()
        self.bits = int(entry['bits'])
        self.shape = tuple(entry['shape'])
        self.count = int(entry['count'])
        self.per_channel = bool(entry['per_channel'])
        self.register_buffer('packed_weight', torch.tensor(list(entry['packed']), dtype=torch.uint8))
        self.register_buffer('weight_scale', entry['scale'].detach().float().clone())
        if source.bias is not None:
            self.register_buffer('bias', source.bias.detach().float().clone())
        else:
            self.bias = None

    def weight_float(self):
        q = unpack_signed(self.packed_weight, self.count, self.bits, self.shape)
        weight = q.to(dtype=self.weight_scale.dtype)
        del q
        if self.per_channel:
            scale = self.weight_scale.reshape((-1,) + (1,) * (len(self.shape) - 1))
        else:
            scale = self.weight_scale
        weight.mul_(scale)  # Avoid creating an extra FP32 multiplication output.
        return weight


class PackedConv2d(PackedWeightBase):
    def __init__(self, source, entry):
        super().__init__(source, entry)
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
    def forward(self, x):
        return F.conv2d(decode(x), self.weight_float(), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class PackedLinear(PackedWeightBase):
    def forward(self, x):
        return F.linear(decode(x), self.weight_float(), self.bias)


class PackedReLU6(nn.Module):
    def __init__(self, bits, scale):
        super().__init__()
        self.bits = int(bits)
        self.scale = float(scale)
        self.last_packed_bytes = 0
        self.last_reference_fp32_bytes = 0
    def forward(self, x):
        x = F.relu6(decode(x))
        maximum = (1 << (self.bits - 1)) - 1
        # Only one FP32 quantization temporary; retain uint8 before bit packing.
        q = x.div(self.scale).round_().clamp_(0, maximum).to(torch.uint8)
        packed = PackedActivation(pack_signed(q, self.bits), x.shape, self.bits, self.scale)
        self.last_packed_bytes = packed.nbytes
        self.last_reference_fp32_bytes = packed.reference_fp32_bytes
        return packed


class PackedAvgPool(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(decode(x), 1)


def residual_forward(self, x):
    # A residual branch must decode its (possibly packed) skip connection.
    out = self.block(x)
    return decode(x) + decode(out) if self.use_residual else out


def model_forward(self, x):
    x = self.features(x)
    x = self.pool(decode(x)).flatten(1)
    return decode(self.classifier(x))


def replace_child(model, dotted, new_module):
    parent = model
    parts = dotted.split('.')
    for part in parts[:-1]:
        parent = parent._modules[part]
    parent._modules[parts[-1]] = new_module


def load_compact_runtime(baseline, filepath, device='cpu', activation_bits=None):
    """Build compact *in-memory* model from a v1 saved compressed bundle.

    Original FP32 Conv/Linear Parameter objects are replaced by uint8 buffers.
    ReLU6 outputs are packed at module boundaries and decoded only when consumed.
    The final Linear logits are fake-quantized to match original evaluation behavior.
    """
    bundle = torch.load(filepath, map_location='cpu', weights_only=True)
    if bundle['format'] != 'manual-ptq-v1':
        raise ValueError('Unsupported packed bundle format')
    model = deepcopy(baseline).cpu().eval()
    model.load_state_dict({**bundle['other_state'], **{
        k: torch.zeros(v['shape'], dtype=torch.float32) for k, v in bundle['weights'].items()
    }}, strict=True)
    for name, module in list(model.named_modules()):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            entry = bundle['weights'][name + '.weight']
            replace_child(model, name, PackedConv2d(module, entry) if isinstance(module, nn.Conv2d)
                          else PackedLinear(module, entry))
    bits = int(activation_bits or bundle['activation_bits'])
    activation_scales = bundle['activation_scales']
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.ReLU6):
            replace_child(model, name, PackedReLU6(bits, activation_scales[name]))
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            replace_child(model, name, PackedAvgPool())
    # Adapt custom residual block to packed skip tensors.
    from Models.MobileNetV2 import InvertedResidual
    for module in model.modules():
        if isinstance(module, InvertedResidual):
            module.forward = MethodType(residual_forward, module)
    model.forward = MethodType(model_forward, model)
    # Earlier pipeline also fake-quantized classifier logits; apply same rounding.
    output_scale = activation_scales.get('classifier.1')
    if output_scale is not None:
        classifier = model.classifier[1]
        old_forward = classifier.forward
        qmax = (1 << (bits - 1)) - 1
        def logits_forward(self, x):
            y = old_forward(x)
            return (y / output_scale).round().clamp(-qmax, qmax) * output_scale
        classifier.forward = MethodType(logits_forward, classifier)
    model = model.to(device).eval()
    model._packed_activation_bits = bits
    return model


def resident_bytes(model):
    """Physical resident parameter and buffer tensor payload (excludes CUDA allocator overhead)."""
    tensors = list(model.parameters()) + list(model.buffers())
    seen = set()
    total = 0
    for t in tensors:
        storage = t.untyped_storage()
        ident = (str(t.device), storage.data_ptr())
        if ident not in seen:
            total += storage.nbytes()
            seen.add(ident)
    return total


def packed_model_breakdown(model):
    weight = scale = auxiliary = 0
    for m in model.modules():
        if isinstance(m, PackedWeightBase):
            weight += m.packed_weight.numel() * m.packed_weight.element_size()
            scale += m.weight_scale.numel() * m.weight_scale.element_size()
            if m.bias is not None:
                auxiliary += m.bias.numel() * m.bias.element_size()
        elif isinstance(m, nn.BatchNorm2d):
            auxiliary += sum(x.numel() * x.element_size() for x in m.state_dict().values())
    return {'resident_packed_weight_bytes': weight, 'resident_weight_scales_bytes': scale,
            'resident_auxiliary_bytes': auxiliary, 'resident_model_tensor_bytes': resident_bytes(model)}