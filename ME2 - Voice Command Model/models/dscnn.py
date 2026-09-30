import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.relu_dw = nn.ReLU(inplace=True)
        
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu_pw = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.relu_dw(self.bn_dw(self.depthwise(x)))
        out = self.relu_pw(self.bn_pw(self.pointwise(out)))
        out = self.dropout(out)
        return out


class DSCNN(nn.Module):
    """
    Depthwise Separable CNN for Keyword / Intent Spotting
    Optimized for ultra-low latency on ARM Cortex-A76 (Raspberry Pi 5).
    """
    def __init__(self, num_classes=20, in_channels=1, channels=(64, 64, 128, 128, 128), dropout=0.1):
        super().__init__()
        
        # Initial standard convolution: reduces frequency and time resolution by 2
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Stack of Depthwise Separable convolutions
        blocks = []
        c_in = channels[0]
        for c_out in channels[1:]:
            blocks.append(DepthwiseSeparableConv(c_in, c_out, stride=1, dropout=dropout))
            c_in = c_out
        self.ds_blocks = nn.Sequential(*blocks)
        
        # Global Average Pooling and Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x):
        # x: (B, 1, Mels=40, Time=201)
        feat = self.init_conv(x)
        feat = self.ds_blocks(feat)
        pooled = self.global_pool(feat)
        flattened = torch.flatten(pooled, 1)
        logits = self.fc(flattened)
        return logits

def get_dscnn(num_classes=20):
    return DSCNN(num_classes=num_classes)
