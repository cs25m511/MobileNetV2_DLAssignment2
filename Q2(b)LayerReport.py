"""Question 2(b): exact compressed and excluded layers for current MobileNetV2."""
import csv
from pathlib import Path
import torch
from torch import nn
from Models.MobileNetV2 import MobileNetV2
from Q2Run import CHECKPOINT, RESULTS


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f'Checkpoint missing: {CHECKPOINT}')
    c = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)['config']
    model = MobileNetV2(num_classes=10, width_mult=c.get('width_mult',1.),
                        dropout=c.get('dropout',.2), batch_norm=c.get('use_batchnorm',True))
    rows=[]
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            rows.append({'layer':name,'class':type(m).__name__,
                         'weight_elements':m.weight.numel(),'weight_quantized':True,
                         'activation_packed_boundary':False,'exception':''})
        elif isinstance(m, nn.ReLU6):
            rows.append({'layer':name,'class':'ReLU6','weight_elements':0,
                         'weight_quantized':False,'activation_packed_boundary':True,
                         'exception':'Activation packed; FP32 transient decode'})
        elif isinstance(m,(nn.BatchNorm2d,nn.AdaptiveAvgPool2d,nn.Dropout)):
            rows.append({'layer':name,'class':type(m).__name__,'weight_elements':0,
                         'weight_quantized':False,'activation_packed_boundary':False,
                         'exception':'FP32 / unquantized operation'})
    RESULTS.mkdir(parents=True,exist_ok=True)
    with (RESULTS/'layer_compression_scope.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print('Quantized weight layers:',sum(r['weight_quantized'] for r in rows))
    print('Packed activation boundaries:',sum(r['activation_packed_boundary'] for r in rows))
    print('Saved',RESULTS/'layer_compression_scope.csv')

if __name__=='__main__':main()
