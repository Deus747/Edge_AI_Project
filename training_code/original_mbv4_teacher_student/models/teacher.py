# ============================================================
# models/teacher.py — MobileNetV4-Conv-Large + DeepLabV3+
#
# SWAP teacher: change BACKBONE_NAME and update
# TEACHER_BB_CHANNELS in config.py to match timm output.
#
# Output contract:
#   {
#     'logits'   : (B, NUM_CLASSES, H, W),
#     'bb_feats' : [s1, s2, s3, s4],
#     'dec_feats': [],   # empty — no decoder KD with LR-ASPP student
#   }
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import config


class ASPPConv(nn.Module):
    """Dilated conv branch of ASPP."""
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class ASPPPooling(nn.Module):
    """Global average pooling branch of ASPP."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        size = x.shape[-2:]
        out  = self.block(x)
        return F.interpolate(out, size=size,
                             mode='bilinear', align_corners=False)


class DeepLabV3PlusHead(nn.Module):
    """
    DeepLabV3+ decoder.
    Stage 4 (high_feat) → ASPP.
    Stage 2 (low_feat)  → low-level skip connection.
    DeepLabV3+ chosen because LR-ASPP shares the same
    ASPP lineage — logit KD signal is maximally informative.
    """
    def __init__(self, low_ch, high_ch, aspp_ch=256,
                 low_out_ch=48, num_classes=7):
        super().__init__()
        self.aspp_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(high_ch, aspp_ch, 1, bias=False),
                nn.BatchNorm2d(aspp_ch),
                nn.ReLU(inplace=True),
            ),
            ASPPConv(high_ch, aspp_ch, dilation=6),
            ASPPConv(high_ch, aspp_ch, dilation=12),
            ASPPConv(high_ch, aspp_ch, dilation=18),
            ASPPPooling(high_ch, aspp_ch),
        ])
        self.aspp_project = nn.Sequential(
            nn.Conv2d(aspp_ch * 5, aspp_ch, 1, bias=False),
            nn.BatchNorm2d(aspp_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch, low_out_ch, 1, bias=False),
            nn.BatchNorm2d(low_out_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(aspp_ch + low_out_ch, aspp_ch, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(aspp_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_ch, aspp_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_ch),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(aspp_ch, num_classes, 1)

    def forward(self, low_feat, high_feat):
        aspp_outs = [conv(high_feat) for conv in self.aspp_convs]
        aspp      = self.aspp_project(torch.cat(aspp_outs, dim=1))
        aspp_up   = F.interpolate(aspp, size=low_feat.shape[-2:],
                                  mode='bilinear', align_corners=False)
        low_out   = self.low_proj(low_feat)
        fused     = self.fuse(torch.cat([aspp_up, low_out], dim=1))
        return self.classifier(fused)


class TeacherModel(nn.Module):
    """
    MobileNetV4-Conv-Large + DeepLabV3+
    SWAP: change BACKBONE_NAME and config.TEACHER_BB_CHANNELS.
    """
    BACKBONE_NAME = 'mobilenetv4_conv_large.e600_r384_in1k'

    def __init__(self, num_classes=None, bb_channels=None):
        super().__init__()
        num_classes = num_classes or config.NUM_CLASSES
        bb_channels = bb_channels or config.TEACHER_BB_CHANNELS

        self.backbone = timm.create_model(
            self.BACKBONE_NAME,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.head = DeepLabV3PlusHead(
            low_ch     = bb_channels[1],   # Stage 2
            high_ch    = bb_channels[3],   # Stage 4
            aspp_ch    = 256,
            low_out_ch = 48,
            num_classes= num_classes,
        )

    def forward(self, x):
        H, W     = x.shape[-2:]
        bb_feats = self.backbone(x)
        logits_s = self.head(bb_feats[1], bb_feats[3])
        logits   = F.interpolate(logits_s, size=(H, W),
                                 mode='bilinear', align_corners=False)
        return {
            'logits'   : logits,
            'bb_feats' : bb_feats,
            'dec_feats': [],
        }


def build_teacher(freeze=True):
    """Instantiate teacher and optionally freeze weights."""
    model = TeacherModel().to(config.DEVICE)
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Teacher: {TeacherModel.BACKBONE_NAME} + DeepLabV3+  "
          f"({n:.1f}M params, frozen={freeze})")
    return model
