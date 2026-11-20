"""Exact, generic bit-packing of signed quantized integer values."""
import numpy as np
import torch

class BitPacker:
    @staticmethod
    def pack(signed_tensor, bits):
        if not 1 <= bits <= 8: raise ValueError("Expected 1..8 bits")
        values = signed_tensor.cpu().reshape(-1).to(torch.int16).numpy().astype(np.int32)
        lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        if values.size and (values.min() < lower or values.max() > upper):
            raise ValueError("Integer outside signed bit range")
        # Two's complement in a fixed-width field; most-significant bits first.
        unsigned = values & ((1 << bits) - 1)
        stream = bytearray((values.size * bits + 7) // 8)
        for i, val in enumerate(unsigned):
            v = int(val)
            for b in range(bits):
                if v & (1 << (bits - 1 - b)):
                    absolute = i * bits + b
                    stream[absolute // 8] |= 1 << (7 - absolute % 8)
        return bytes(stream)

    @staticmethod
    def unpack(payload, count, bits, shape):
        if not 1 <= bits <= 8: raise ValueError("Expected 1..8 bits")
        if len(payload) != (count * bits + 7) // 8: raise ValueError("Invalid packed byte count")
        values = []
        for i in range(count):
            value = 0
            for b in range(bits):
                absolute = i * bits + b
                bit = (payload[absolute // 8] >> (7 - absolute % 8)) & 1
                value = (value << 1) | bit
            if value & (1 << (bits - 1)): value -= 1 << bits
            values.append(value)
        return torch.tensor(values, dtype=torch.int16).reshape(shape)
