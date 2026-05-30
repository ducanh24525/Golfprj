import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import numpy as np

# ================= CBAM =================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        max = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg + max)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, max], dim=1)
        return x * self.sigmoid(self.conv(attn))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ================= EVENT DETECTOR =================
class EventDetector(nn.Module):
    def __init__(
        self,
        pretrain=True,
        width_mult=1.0,

        # RGB branch
        rgb_lstm_hidden=256,
        rgb_lstm_layers=1,
        rgb_gru_hidden=256,

        # Pose branch
        pose_dim=26,
        pose_lstm_hidden=128,
        pose_lstm_layers=1,
        pose_gru_hidden=128,

        bidirectional=True,
        dropout=True,
        num_classes=9
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.rgb_lstm_layers = rgb_lstm_layers
        self.pose_lstm_layers = pose_lstm_layers

        # ============ RGB BRANCH ============
        net = mobilenet_v2(pretrained=pretrain, width_mult=width_mult)
        self.cnn = net.features
        cnn_out_dim = net.last_channel  # 1280

        self.cbam = CBAM(cnn_out_dim)
        self.rgb_drop = nn.Dropout(0.3) if dropout else nn.Identity()

        self.rgb_lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=rgb_lstm_hidden,
            num_layers=rgb_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if rgb_lstm_layers > 1 else 0.0
        )

        rgb_lstm_out = 2 * rgb_lstm_hidden if bidirectional else rgb_lstm_hidden

        self.rgb_gru = nn.GRU(
            input_size=rgb_lstm_out,
            hidden_size=rgb_gru_hidden,
            batch_first=True
        )

        rgb_out_dim = rgb_gru_hidden

        # ============ POSE BRANCH ============
        self.pose_lstm = nn.LSTM(
            input_size=pose_dim,
            hidden_size=pose_lstm_hidden,
            num_layers=pose_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if pose_lstm_layers > 1 else 0.0
        )

        pose_lstm_out = 2 * pose_lstm_hidden if bidirectional else pose_lstm_hidden

        self.pose_gru = nn.GRU(
            input_size=pose_lstm_out,
            hidden_size=pose_gru_hidden,
            batch_first=True
        )

        pose_out_dim = pose_gru_hidden

        # ============ CROSS-ATTENTION ============
        self.pose_proj = nn.Linear(pose_out_dim, rgb_out_dim)

        # ============ FUSION ============
        self.classifier = nn.Linear(
            rgb_out_dim + pose_out_dim,
            num_classes
        )

    def _init_lstm_hidden(self, layers, hidden, batch, device):
        num_dir = 2 if self.bidirectional else 1
        h0 = torch.zeros(num_dir * layers, batch, hidden, device=device)
        c0 = torch.zeros_like(h0)
        return h0, c0

    def forward(self, images, poses):
        """
        images: (B, T, 3, H, W)
        poses : (B, T, J, C)
        """
        B, T, C, H, W = images.size()
        device = images.device

        # ===== RGB =====
        x = images.view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.cbam(x)
        x = x.mean(dim=[2, 3])          # GAP
        x = self.rgb_drop(x)
        x = x.view(B, T, -1)

        h0, c0 = self._init_lstm_hidden(
            self.rgb_lstm_layers,
            self.rgb_lstm.hidden_size,
            B,
            device
        )
        x, _ = self.rgb_lstm(x, (h0, c0))   # LSTM first
        x, _ = self.rgb_gru(x)              # GRU after

        # ===== POSE =====
        poses = poses[:, :, :, :2].reshape(B, T, 26)

        h0p, c0p = self._init_lstm_hidden(
            self.pose_lstm_layers,
            self.pose_lstm.hidden_size,
            B,
            device
        )
        p, _ = self.pose_lstm(poses, (h0p, c0p))  # LSTM first
        p, _ = self.pose_gru(p)                   # GRU after

        # ===== CROSS-ATTENTION (RGB ← Pose) =====
        p_proj = self.pose_proj(p)  # (B,T,Drgb)
        D = x.size(-1)

        attn = torch.softmax(
            torch.bmm(x, p_proj.transpose(1, 2)) / (D ** 0.5),
            dim=-1
        )
        x = x + torch.bmm(attn, p_proj)   # residual

        # ===== FUSION =====
        feat = torch.cat([x, p], dim=-1)
        out = self.classifier(feat)

        return out.view(B * T, -1)

# Transforms
class NormalizePose(object):
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.LEFT_HIP = 7
        self.RIGHT_HIP = 8
        self.LEFT_SHOULDER = 1
        self.RIGHT_SHOULDER = 2

    def __call__(self, sample):
        images, poses = sample['images'], sample['poses']
        poses = poses.copy()  # (T, 13, 3)

        # Root-centered
        hip_center = (poses[:, self.LEFT_HIP, :2] + poses[:, self.RIGHT_HIP, :2]) / 2.0
        poses[:, :, :2] -= hip_center[:, None, :]

        # Scale by torso length
        shoulder_center = (poses[:, self.LEFT_SHOULDER, :2] + poses[:, self.RIGHT_SHOULDER, :2]) / 2.0
        torso = np.linalg.norm(shoulder_center, axis=1, keepdims=True)
        torso = np.maximum(torso, self.eps)
        poses[:, :, :2] /= torso[:, None, :]

        return {'images': images, 'poses': poses}

class ToTensor(object):
    def __call__(self, sample):
        images, poses = sample['images'], sample['poses']
        images = images.transpose((0, 3, 1, 2))  # HWC -> CHW
        images = torch.from_numpy(images).float().div(255.)
        poses = torch.from_numpy(poses).float()
        return {'images': images, 'poses': poses}

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, poses = sample['images'], sample['poses']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'poses': poses}
