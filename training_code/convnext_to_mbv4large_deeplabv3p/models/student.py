import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import config


class ASPPConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPPPooling(nn.Module):
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
        out = self.block(x)
        return F.interpolate(out, size=size, mode='bilinear', align_corners=False)


class DeepLabV3PlusHeadStudent(nn.Module):
    def __init__(self, low_ch, high_ch, num_classes=7, aspp_ch=256, low_out_ch=48):
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
            nn.Conv2d(aspp_ch + low_out_ch, aspp_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_ch, aspp_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_ch),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(aspp_ch, num_classes, 1)
        self.low_logits = nn.Conv2d(low_out_ch, num_classes, 1)

        # Adapters to match existing KD decoder bridge expectations.
        self.kd_high_128 = nn.Conv2d(aspp_ch, 128, 1, bias=False)
        self.kd_mid_128 = nn.Conv2d(aspp_ch, 128, 1, bias=False)

    def forward(self, low_feat, high_feat):
        aspp_outs = [conv(high_feat) for conv in self.aspp_convs]
        aspp = self.aspp_project(torch.cat(aspp_outs, dim=1))          # 256 @ 1/16
        aspp_up = F.interpolate(aspp, size=low_feat.shape[-2:], mode='bilinear', align_corners=False)
        low_out = self.low_proj(low_feat)                               # 48 @ 1/4
        fused = self.fuse(torch.cat([aspp_up, low_out], dim=1))         # 256 @ 1/4
        logits_small = self.classifier(fused)                           # C @ 1/4

        low_logits = self.low_logits(low_out)                           # C @ 1/4
        dec_feats = [
            self.kd_high_128(aspp),   # index 0
            self.kd_high_128(aspp),   # index 1 placeholder
            self.kd_mid_128(fused),   # index 2
            logits_small,             # index 3
            low_logits,               # index 4
        ]
        return logits_small, dec_feats


class StudentModel(nn.Module):
    BACKBONE_NAME = 'mobilenetv4_conv_large.e600_r384_in1k'

    def __init__(self, num_classes=None, bb_channels=None, is_export=False, pretrained=True):
        super().__init__()
        self.is_export = is_export
        num_classes = num_classes or config.NUM_CLASSES
        bb_channels = bb_channels or config.STUDENT_BB_CHANNELS

        self.backbone = timm.create_model(
            self.BACKBONE_NAME,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.head = DeepLabV3PlusHeadStudent(
            low_ch=bb_channels[1],
            high_ch=bb_channels[3],
            num_classes=num_classes,
            aspp_ch=256,
            low_out_ch=48,
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        bb_feats = self.backbone(x)
        logits_small, dec_feats = self.head(bb_feats[1], bb_feats[3])
        logits = F.interpolate(logits_small, size=(h, w), mode='bilinear', align_corners=False)

        if self.is_export:
            return logits

        return {
            'logits': logits,
            'bb_feats': bb_feats,
            'dec_feats': dec_feats,
        }


def build_student(is_export=False):
    return StudentModel(is_export=is_export, pretrained=True).to(config.DEVICE)
