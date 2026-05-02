import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


CLASS_NAMES = [f"class_{i}" for i in range(16)]


class LRASPPHead(nn.Module):
    def __init__(self, low_ch=32, high_ch=96, inter_ch=128, num_classes=16):
        super().__init__()
        self.high_conv = nn.Sequential(
            nn.Conv2d(high_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True),
        )
        self.gap_gate = nn.Sequential(
            nn.AvgPool2d(kernel_size=32, stride=32),
            nn.Conv2d(high_ch, inter_ch, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.low_conv = nn.Conv2d(low_ch, num_classes, kernel_size=1, bias=False)
        self.classifier = nn.Conv2d(inter_ch, num_classes, kernel_size=1, bias=False)

    def forward(self, low_feat, high_feat):
        high_out = self.high_conv(high_feat)
        gate = self.gap_gate(high_feat)
        gated = high_out * gate
        gated_up = F.interpolate(gated, size=(128, 128), mode="bilinear", align_corners=False)

        high_logits = self.classifier(gated_up)
        low_logits = self.low_conv(low_feat)
        return high_logits + low_logits


class Model1SegmentationModel(nn.Module):
    BACKBONE_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"

    def __init__(self, num_classes=16, bb_channels=None, is_export=False):
        super().__init__()
        self.is_export = is_export

        if bb_channels is None:
            bb_channels = [32, 32, 64, 96]

        self.backbone = timm.create_model(
            self.BACKBONE_NAME,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.head = LRASPPHead(
            low_ch=bb_channels[1],
            high_ch=bb_channels[3],
            inter_ch=128,
            num_classes=num_classes,
        )

    def forward(self, x):
        bb_feats = self.backbone(x)
        logits_s = self.head(bb_feats[1], bb_feats[3])
        logits = F.interpolate(logits_s, size=(512, 512), mode="bilinear", align_corners=False)

        if self.is_export:
            return logits

        return {
            "logits": logits,
            "bb_feats": bb_feats,
            "dec_feats": [],
        }


def build_model(num_classes=16, bb_channels=None, is_export=False, device="cpu"):
    model = Model1SegmentationModel(
        num_classes=num_classes,
        bb_channels=bb_channels,
        is_export=is_export,
    ).to(device)
    return model
