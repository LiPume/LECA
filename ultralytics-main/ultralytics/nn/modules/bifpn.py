import torch
import torch.nn as nn
import torch.nn.functional as F


# 基础模块定义 (来自你的代码)
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.bn(self.pointwise(self.depthwise(x))))
        return x


# BiFPN融合模块 (来自你的代码)
class BiFPNBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv3 = SeparableConv2d(channels, channels)
        self.conv4 = SeparableConv2d(channels, channels)
        self.conv5 = SeparableConv2d(channels, channels)

        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.epsilon = 1e-4

    def forward(self, P3, P4, P5):
        # Top-down
        w = F.relu(self.w1)
        w = w / (w.sum() + self.epsilon)
        P4_td = self.conv4(w[0] * P4 + w[1] * F.interpolate(P5, scale_factor=2))

        # Bottom-up
        w = F.relu(self.w2)
        w = w / (w.sum() + self.epsilon)
        P3_out = self.conv3(w[0] * P3 + w[1] * F.interpolate(P4_td, scale_factor=2))
        P5_out = self.conv5(w[2] * P5 + w[1] * F.max_pool2d(P4_td, kernel_size=2))

        return P3_out, P4_td, P5_out


# 整体Neck模块封装 (来自你的代码 + 关键修改)
class BiFPN(nn.Module):
    def __init__(self, in_channels_list=[128, 256, 512], out_channels=256, num_repeats=2):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        self.bifpn_blocks = nn.Sequential(*[BiFPNBlock(out_channels) for _ in range(num_repeats)])

        # (可选) 增加 Dropout，如你所想
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features):
        P3, P4, P5 = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        for block in self.bifpn_blocks:
            P3, P4, P5 = block(P3, P4, P5)

        # (可选) 应用 Dropout
        P3 = self.dropout(P3)
        P4 = self.dropout(P4)
        P5 = self.dropout(P5)

        # [!!!] 关键修改：
        # YOLO 的 Detect Head 需要一个 *列表* (List) 作为输入，而不是元组 (Tuple)
        # 因为它需要在列表上执行 x[i] = ... 的修改操作
        return [P3, P4, P5]

# test