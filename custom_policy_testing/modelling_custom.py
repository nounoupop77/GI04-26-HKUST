import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from torchvision.models import resnet18, ResNet18_Weights


class SpatialSoftmax(nn.Module):
    """Per-channel soft-argmax over a 7x7 feature map -> [B, 2C] in [-1, 1]."""
    def __init__(self, height=7, width=7, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, width),
            torch.linspace(-1.0, 1.0, height),
            indexing="xy",
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))

    def forward(self, feat):
        B, C, H, W = feat.shape
        assert (H, W) == (7, 7), f"SpatialSoftmax expected 7x7, got {H}x{W}"
        flat = feat.reshape(B, C, -1)
        attn = F.softmax(flat / self.temperature, dim=-1)
        x = (attn * self.pos_x).sum(dim=-1)
        y = (attn * self.pos_y).sum(dim=-1)
        return torch.cat([x, y], dim=1)


def _freeze_bn(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

# ---------------------------------------------------------------------------
# Keypoint encoder: 4-dim (block_xy, target_xy) -> embedding, with modality dropout
# ---------------------------------------------------------------------------
class KeypointEncoder(nn.Module):
    """MLP that embeds the 4-dim detection vector.

    During training with probability `dropout_prob`, the whole vector is
    zeroed for that sample. At eval time it is always used.
    Set dropout_prob = 0.0 to always use, ~0.8 for ~20% kept.
    """
    def __init__(self, in_dim: int = 8, emb_dim: int = 64, dropout_prob: float = 0.2):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, emb_dim),
            nn.ReLU(),
        )

    def forward(self, kp, kp_presence=None):
        # kp: [B, 4], kp_presence: [B, 4]
        if kp_presence is None:
            kp_presence = torch.ones_like(kp)

        x = torch.cat([kp, kp_presence], dim=1)   # [B, 8]

        if self.training and self.dropout_prob > 0.0:
            B = x.shape[0]
            keep = (torch.rand(B, 1, device=x.device) > self.dropout_prob).float()
            x = x * keep   # zeros out both coords AND presence flags together
        return self.net(x)


class CustomPolicy(nn.Module):
    name = "custom"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_frames         = getattr(config, "num_frames", 3)
        self.state_dim          = getattr(config, "state_dim", 6)
        self.action_dim         = getattr(config, "action_dim", 6)
        self.action_chunk_size  = getattr(config, "action_chunk_size", 8)
        self.bn_train_in_layer4 = getattr(config, "bn_train_in_layer4", False)
        self.feature_dim        = 512
        self.keypoint_dim       = self.feature_dim * 2   # 1024

        kp_in   = getattr(config, "keypoint_dim", 8)
        kp_emb  = getattr(config, "keypoint_emb_dim", 64)
        kp_drop = getattr(config, "keypoint_dropout_prob", 0.2)

        self.encoder_top   = self._make_resnet18_encoder()
        self.encoder_wrist = self._make_resnet18_encoder()
        self.ssm_top       = SpatialSoftmax(7, 7)
        self.ssm_wrist     = SpatialSoftmax(7, 7)

        self.kp_encoder = KeypointEncoder(kp_in, kp_emb, kp_drop)

        combined_dim = self.keypoint_dim * 2 + self.state_dim + kp_emb
        out_dim = self.action_dim * self.action_chunk_size

        self.fc = nn.Sequential(
            nn.Linear(combined_dim, config.fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(config.fc_hidden_dim, out_dim),
        )

    def _make_resnet18_encoder(self):
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
        for p in model.parameters():
            p.requires_grad = False
        _freeze_bn(model)
        return model

    def unfreeze_last_block(self):
        for encoder in (self.encoder_top, self.encoder_wrist):
            for p in encoder.layer4.parameters():
                p.requires_grad = True
            for m in encoder.layer4.modules():
                if isinstance(m, nn.BatchNorm2d):
                    if self.bn_train_in_layer4:
                        m.train()
                    else:
                        m.eval()
                        for p in m.parameters():
                            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for enc in (self.encoder_top, self.encoder_wrist):
                for name, m in enc.named_modules():
                    if isinstance(m, nn.BatchNorm2d):
                        in_layer4 = name.startswith("layer4")
                        if in_layer4 and self.bn_train_in_layer4 and \
                           any(p.requires_grad for p in m.parameters()):
                            m.train()
                        else:
                            m.eval()
                for blk in ("conv1", "bn1", "layer1", "layer2", "layer3"):
                    getattr(enc, blk, nn.Identity()).eval()
        return self

    def _run_backbone(self, encoder, x):
        x = encoder.conv1(x); x = encoder.bn1(x); x = encoder.relu(x)
        x = encoder.maxpool(x)
        x = encoder.layer1(x); x = encoder.layer2(x)
        x = encoder.layer3(x); x = encoder.layer4(x)
        return x

    def _encode_frames(self, encoder, ssm, imgs):
        B, N = imgs.shape[0], imgs.shape[1]
        x = imgs.reshape(B * N, *imgs.shape[2:])
        has_grad = any(p.requires_grad for p in encoder.parameters())
        if not has_grad:
            with torch.no_grad():
                feat = self._run_backbone(encoder, x)
        else:
            feat = self._run_backbone(encoder, x)
        kp = ssm(feat)
        return kp.view(B, N, -1)

    def forward(self, batch: dict[str, Any]):
        img_front = batch["observation.images.front"]      # [B, N, 3, H, W]
        img_side  = batch["observation.images.side"]       # [B, N, 3, H, W]
        state     = batch["observation.state"]             # [B, 6]
        keypoint  = batch["observation.keypoint"]          # [B, 4]
        kp_presence = batch.get("observation.keypoint_presence", None)

        kp_front = self._encode_frames(self.encoder_top,   self.ssm_top,   img_front)[:,-1]
        kp_side  = self._encode_frames(self.encoder_wrist, self.ssm_wrist, img_side)[:,-1]

        kp_emb  = self.kp_encoder(keypoint,kp_presence)

        combined = torch.cat([kp_front, kp_side, state, kp_emb], dim=1)
        out = self.fc(combined)
        return out.view(-1, self.action_chunk_size, self.action_dim)

    def select_action(self, batch: dict[str, Any]):
        self.eval()
        with torch.no_grad():
            return self.forward(batch)