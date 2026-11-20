"""MobileNetV2 implemented using primitive PyTorch layers (no model library)."""
from torch import nn


def make_divisible(value, divisor=8):
    result = max(divisor, int(value + divisor / 2) // divisor * divisor)
    return result + divisor if result < 0.9 * value else result


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 groups=1, batch_norm=True):
        padding = (kernel_size - 1) // 2
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                            padding, groups=groups, bias=not batch_norm)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, batch_norm=True):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        hidden_channels = int(round(in_channels * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_channels, kernel_size=1,
                                     batch_norm=batch_norm))
        layers.extend([
            ConvBNReLU(hidden_channels, hidden_channels, stride=stride,
                       groups=hidden_channels, batch_norm=batch_norm),
            nn.Conv2d(hidden_channels, out_channels, 1, bias=not batch_norm),
        ])
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)
        self.use_residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_residual else out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, dropout=0.2, batch_norm=True):
        super().__init__()
        if width_mult <= 0 or not 0 <= dropout < 1:
            raise ValueError("width_mult > 0 and dropout in [0,1) required")
        stages = [(1,16,1,1), (6,24,2,2), (6,32,3,2), (6,64,4,2),
                  (6,96,3,1), (6,160,3,2), (6,320,1,1)]
        in_ch = make_divisible(32 * width_mult)
        layers = [ConvBNReLU(3, in_ch, stride=1, batch_norm=batch_norm)]
        for expansion, channels, repeats, stride in stages:
            out_ch = make_divisible(channels * width_mult)
            for i in range(repeats):
                layers.append(InvertedResidual(in_ch, out_ch,
                              stride if i == 0 else 1, expansion, batch_norm=batch_norm))
                in_ch = out_ch
        final_ch = make_divisible(1280 * width_mult) if width_mult > 1 else 1280
        layers.append(ConvBNReLU(in_ch, final_ch, kernel_size=1, batch_norm=batch_norm))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(final_ch, num_classes))
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))
