import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    """Efficient Channel Attention (ECA)"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        k = k if k % 2 else k + 1  # kernel size must be odd
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)          # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)   # [B, 1, C]
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os  # [必须] 导入 os 模块

# class LECA(nn.Module):
#     """
#     LECA-Ultimate: 
#     - softplus variance suppression
#     - weak feature recovery
#     - ECA calibration
#     - brightness correction
    
#     [Updated]: Supports training control via Environment Variables
#     """

#     def __init__(self, channels, gamma=2, b=1):
#         super().__init__()

#         # ----- ECA 部分 (基础结构) -----
#         t = int(abs((math.log2(channels) + b) / gamma))
#         k = t if t % 2 else t + 1
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1)//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#         # ----- ULECA 参数 (可学习参数) -----
#         # 即使被关闭，参数依然会被创建，但在反向传播中梯度为0，不会影响训练

#         self.beta = nn.Parameter(torch.tensor(1e-5)) 
#         self.alpha = nn.Parameter(torch.tensor(1e-5))
#         self.gamma = 1e-5 # 注意这里如果是标量，保持浮点数即可；如果是Parameter则用torch.tensor(1e-5)

#         # ----- [核心修改] 从环境变量读取训练配置 -----
#         # 逻辑：
#         # 1. os.environ.get 获取环境变量，默认为 '1' (开启)
#         # 2. 如果是 '1' 则为 True，否则为 False
#         # 这样设计的好处是：如果你正常跑训练，不设环境变量，它默认就是 Full ULECA
#         self.use_var = os.environ.get('ULECA_USE_VAR', '1') == '1'
#         self.use_rec = os.environ.get('ULECA_USE_REC', '1') == '1'
#         self.use_bri = os.environ.get('ULECA_USE_BRI', '1') == '1'
        
#         # (可选) 打印一下当前模块的状态，方便调试看日志
#         # print(f"Init LECA | Var:{self.use_var} Rec:{self.use_rec} Bri:{self.use_bri}")

#     def forward(self, x):
#         # 1. ----- ECA (Base, 永远开启) -----
#         # 如果把下面三个都关了，这里就等价于标准的 ECA
#         avg = self.avg_pool(x)
#         y = avg.squeeze(-1).transpose(-1, -2)
#         y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
#         w_eca = self.sigmoid(y)
        
#         w_final = w_eca

#         # 2. ----- Variance Suppression -----
#         # 使用 self.use_var (初始化时已确定的布尔值)
#         if self.use_var:
#             mean = x.mean(dim=(2,3), keepdim=True)
#             mean2 = (x * x).mean(dim=(2,3), keepdim=True)
#             var = mean2 - mean * mean
#             noise = F.softplus(var)
#             w_sup = 1 / (1 + self.beta * noise)
#             w_final = w_final * w_sup

#         # 3. ----- Weak Feature Recovery -----
#         if self.use_rec:
#             # 为了效率，如果上面没算 mean，这里需要算一下
#             if not self.use_var:
#                 mean = x.mean(dim=(2,3), keepdim=True)
            
#             low = torch.sigmoid(-mean)
#             w_rec = 1 + self.alpha * low
#             w_final = w_final * w_rec

#         # 4. ----- Brightness Correction -----
#         if self.use_bri:
#             local_mean = torch.mean(x, dim=1, keepdim=True)
#             corr = torch.sigmoid(-local_mean).mean((2,3), keepdim=True)
#             w_bri = 1 + self.gamma * corr
#             w_final = w_final * w_bri

#         return x * w_final
class LECA(nn.Module):
    # ==========================================
    # 1. 定义类变量 (作为全局控制开关)
    # ==========================================
    # 这些是默认值，外部脚本可以通过 LECA.INIT_ALPHA = 0.01 来修改它
    INIT_ALPHA = 0.02
    INIT_BETA = 0.04
    INIT_GAMMA = 0.01

    def __init__(self, channels):
        super().__init__()

        # ECA 部分 (保持不变)
        gamma_eca = 2
        b_eca = 1
        t = int(abs((math.log2(channels) + b_eca) / gamma_eca))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # ==========================================
        # 2. 使用类变量初始化 Parameter
        # ==========================================
        # 注意：这里引用的是 self.__class__.INIT_XXX，或者是 LECA.INIT_XXX
        # 这样我们在外部修改类变量后，这里实例化的值就会变
        self.beta = nn.Parameter(torch.tensor(float(LECA.INIT_BETA)))
        self.alpha = nn.Parameter(torch.tensor(float(LECA.INIT_ALPHA)))
        self.gamma = nn.Parameter(torch.tensor(float(LECA.INIT_GAMMA)))

    def forward(self, x):
        # ... (你的 forward 代码保持完全不变) ...
        # (为了节省篇幅，这里省略 forward 内容，和之前一样即可)
        B, C, H, W = x.size()
        avg = self.avg_pool(x)
        y = avg.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        w_eca = self.sigmoid(y)

        mean = x.mean(dim=(2,3), keepdim=True)
        mean2 = (x * x).mean(dim=(2,3), keepdim=True)
        var = mean2 - mean * mean
        noise = F.softplus(var)
        w_sup = 1 / (1 + self.beta * noise)

        low = torch.sigmoid(-mean)
        w_rec = 1 + self.alpha * low

        local_mean = torch.mean(x, dim=1, keepdim=True)
        corr = torch.sigmoid(-local_mean).mean((2,3), keepdim=True)
        w_bri = 1 + self.gamma * corr

        w = w_eca * w_sup * w_rec * w_bri
        return x * w
class LUECA(nn.Module):
    """
    LECA-Ultimate: 
    - softplus variance suppression
    - weak feature recovery
    - ECA calibration
    - brightness correction
    """

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()

        # ECA
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # hyper
        self.beta = nn.Parameter(torch.tensor(0.04))  # 降低一点（更稳）
        self.alpha = nn.Parameter(torch.tensor(0.02)) # 弱恢复强度
        self.gamma = nn.Parameter(torch.tensor(0.01))  # brightness correction

    def forward(self, x):
        B, C, H, W = x.size()

        # ----- ECA -----
        avg = self.avg_pool(x)
        y = avg.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        w_eca = self.sigmoid(y)

        # ----- variance suppression (softplus) -----
        mean = x.mean(dim=(2,3), keepdim=True)
        mean2 = (x * x).mean(dim=(2,3), keepdim=True)
        var = mean2 - mean * mean
        noise = F.softplus(var)
        w_sup = 1 / (1 + self.beta * noise)

        # ----- weak target recovery -----
        low = torch.sigmoid(-mean)
        w_rec = 1 + self.alpha * low

        # ----- brightness correction (提升 recall) -----
        local_mean = torch.mean(x, dim=1, keepdim=True)
        corr = torch.sigmoid(-local_mean).mean((2,3), keepdim=True)
        w_bri = 1 + self.gamma * corr

        # ----- combine -----
        w = w_eca * w_sup * w_rec * w_bri

        return x * w

class AdaptiveULECA(nn.Module):
    """
    Adaptive ULECA (Stable Version)
    使用最稳健的 1/(1+x) 公式，防止特征崩塌。
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        
        # 1. ECA Base
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 2. Weight Predictor
        # 输出初始化为负值，保证初始 alpha/beta/gamma ≈ 0 (Identity Mapping)
        reduction = max(8, channels // 16)
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, 2, 1),  # 注意输出维度改为2，因为不需要Bri
            nn.Sigmoid()
        )
        # [关键] 初始化 bias = -5.0，让初始权重极小，模型从 ECA 平滑过渡
        nn.init.constant_(self.weight_predictor[-2].bias, -5.0)

        # 3. 环境变量
        self.use_var = os.environ.get('A_ULECA_VAR', '1') == '1'
        self.use_rec = os.environ.get('A_ULECA_REC', '1') == '1'

    def forward(self, x):
        # --- A. ECA ---
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        w_eca = self.sigmoid(y).expand_as(x)

        # --- B. 统计因子 ---
        # 1. 方差 (Variance) -> 噪声强度
        # 使用 torch.var 更稳定
        var = torch.var(x, dim=(2,3), keepdim=True)
        noise = F.softplus(var)
        # 归一化：除以通道均值，避免数值过大
        term_sup = noise / (noise.mean(dim=1, keepdim=True) + 1e-6)

        # 2. 恢复 (Recovery) -> 弱特征强度
        mean = x.mean(dim=(2,3), keepdim=True)
        term_rec = torch.sigmoid(-mean)

        # --- C. 自适应参数 ---
        pred_weights = self.weight_predictor(x)
        alpha = pred_weights[:, 0:1, :, :] * (1.0 if self.use_var else 0.0)
        beta  = pred_weights[:, 1:2, :, :] * (1.0 if self.use_rec else 0.0)

        # --- D. 融合 (回归经典公式) ---
        # [核心修正] 使用 1 / (1 + x) 形式，永远不会小于 0，非常稳定
        suppression = 1.0 / (1.0 + alpha * term_sup)
        
        # 增强项维持原样 (1 + x)
        recovery    = 1.0 + beta * term_rec
        
        w_final = w_eca * suppression * recovery

        return x * w_final

class CascadeULECA(nn.Module):
    """
    Cascade ULECA:
    Uses fixed hyperparameters and sequential processing to ensure stability.
    Pipeline: Input -> Variance Suppression -> Weak Recovery -> Brightness Correction -> ECA -> Output
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        
        # 1. ECA Base (Standard)
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 2. Fixed Hyperparameters (Deterministic!)
        # 使用经验值，固定为常量，确保 100% 可复现
        self.alpha = 0.04  # Variance Suppression strength
        self.beta  = 0.02  # Weak Recovery strength
        self.gamma = 0.01  # Brightness Correction strength

        # 3. Env Control (Training Ablation Switches)
        # 默认全部开启 ('1')
        self.use_var = os.environ.get('C_ULECA_VAR', '1') == '1'
        self.use_rec = os.environ.get('C_ULECA_REC', '1') == '1'
        self.use_bri = os.environ.get('C_ULECA_BRI', '1') == '1'

    def forward(self, x):
        # input x: [B, C, H, W]
        
        # --- Stage 1: Variance Suppression (Pre-cleaning) ---
        if self.use_var:
            mean = x.mean(dim=(2,3), keepdim=True)
            mean2 = (x * x).mean(dim=(2,3), keepdim=True)
            var = mean2 - mean * mean
            noise = F.softplus(var)
            
            # 稳健抑制公式: 1 / (1 + alpha * noise)
            w_sup = 1.0 / (1.0 + self.alpha * noise)
            x = x * w_sup  # 直接修改特征图，传给下一级

        # --- Stage 2: Weak Feature Recovery ---
        if self.use_rec:
            # 级联优势：这里的 mean 是基于已经降噪过的特征图计算的
            mean = x.mean(dim=(2,3), keepdim=True)
            low = torch.sigmoid(-mean)
            
            w_rec = 1.0 + self.beta * low
            x = x * w_rec

        # --- Stage 3: Brightness Correction (Optional) ---
        if self.use_bri:
            local_mean = torch.mean(x, dim=1, keepdim=True)
            corr = torch.sigmoid(-local_mean).mean(dim=(2,3), keepdim=True)
            
            w_bri = 1.0 + self.gamma * corr
            x = x * w_bri

        # --- Stage 4: ECA (Fine-tuning) ---
        # ECA 现在看到的是经过 Var/Rec/Bri 处理过的"干净"特征图
        # 它的通道注意力权重会更加精准
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        w_eca = self.sigmoid(y).expand_as(x)
        
        return x * w_eca

