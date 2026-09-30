import torch
import torch.nn as nn

class SubSpectralNorm(nn.Module):
    def __init__(self, channels, sub_bands=4):
        super().__init__()
        self.sub_bands = sub_bands
        self.bn = nn.BatchNorm2d(channels * sub_bands)

    def forward(self, x):
        # x: (B, C, F, T)
        b, c, f, t = x.size()
        assert f % self.sub_bands == 0, f"Frequency {f} not divisible by sub_bands {self.sub_bands}"
        f_sub = f // self.sub_bands
        x = x.view(b, c * self.sub_bands, f_sub, t)
        x = self.bn(x)
        x = x.view(b, c, f, t)
        return x


class BroadcastResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), dropout=0.1):
        super().__init__()
        self.stride = stride
        
        # Pointwise Conv 1x1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Depthwise Conv 3x3 across Time and Frequency
        self.dw_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels,
            bias=False
        )
        self.bn_dw = nn.BatchNorm2d(out_channels)
        
        # Frequency pooling branch (broadcast feature along frequency)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.broadcast_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Shortcut connection
        if in_channels != out_channels or stride != (1, 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn_dw(self.dw_conv(out))
        
        # Broadcast temporal context
        bc = self.freq_pool(out) # (B, C, 1, T)
        # Broadcast temporal context across frequency dimension
        bc = out.mean(dim=2, keepdim=True) # (B, C, 1, T)
        bc = self.broadcast_conv(bc)
        out = out + bc
        
        out = self.relu(out)
        out = self.dropout(out)
        out = out + res
        return out


class BCResNet(nn.Module):
    """
    Broadcasted Residual Network (BC-ResNet-1 / BC-ResNet-3 variant)
    Optimized for acoustic keyword spotting with minimal parameter footprint.
    """
    def __init__(self, num_classes=20, in_channels=1, base_channels=32, dropout=0.1):
        super().__init__()
        
        # Initial Conv
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            BroadcastResBlock(base_channels, base_channels, stride=(1, 1), dropout=dropout),
            BroadcastResBlock(base_channels, base_channels, stride=(1, 1), dropout=dropout)
        )
        
        # Stage 2: Stride in time and frequency
        self.stage2 = nn.Sequential(
            BroadcastResBlock(base_channels, base_channels * 2, stride=(2, 2), dropout=dropout),
            BroadcastResBlock(base_channels * 2, base_channels * 2, stride=(1, 1), dropout=dropout)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            BroadcastResBlock(base_channels * 2, base_channels * 3, stride=(2, 2), dropout=dropout),
            BroadcastResBlock(base_channels * 3, base_channels * 3, stride=(1, 1), dropout=dropout)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 3, num_classes)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        logits = self.fc(out)
        return logits

def get_bcresnet(num_classes=20):
    return BCResNet(num_classes=num_classes)
