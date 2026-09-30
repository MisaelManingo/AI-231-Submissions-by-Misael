import torch
import torch.nn as nn

class MicroDSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(
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
        
        self.pw = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu_pw = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu_dw(self.bn_dw(self.dw(x)))
        x = self.relu_pw(self.bn_pw(self.pw(x)))
        return x


class MicroWakeNet(nn.Module):
    """
    Ultra-lightweight Wake Word Spotter for 'Hey Raspberry'.
    Evaluates a 1.0s window in <2ms on RPi5 CPU.
    """
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        self.blocks = nn.Sequential(
            MicroDSConv(24, 32, stride=(1, 1)),
            MicroDSConv(32, 48, stride=(2, 2)),
            MicroDSConv(48, 64, stride=(1, 1))
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, 1, 40, 101)
        x = self.init_conv(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

def get_micro_wakenet():
    return MicroWakeNet()
