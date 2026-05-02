import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import config


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class PPM(nn.Module):
    """Pyramid Pooling Module for UPerNet."""
    def __init__(self, in_ch, out_ch, bins=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_sz),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for bin_sz in bins
        ])
        self.bottleneck = ConvBNReLU(in_ch + len(bins) * out_ch, in_ch, k=3, p=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [x]
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
            outs.append(y)
        return self.bottleneck(torch.cat(outs, dim=1))


class UPerHead(nn.Module):
    """
    UPerNet-style decoder:
      - PPM on C5
      - top-down FPN fusion to P2
      - multi-scale fusion and classifier
    """
    def __init__(self, in_channels, fpn_ch=256, num_classes=7):
        super().__init__()
        c2, c3, c4, c5 = in_channels

        self.ppm = PPM(c5, out_ch=fpn_ch // 4)

        self.lateral_c2 = nn.Conv2d(c2, fpn_ch, 1, bias=False)
        self.lateral_c3 = nn.Conv2d(c3, fpn_ch, 1, bias=False)
        self.lateral_c4 = nn.Conv2d(c4, fpn_ch, 1, bias=False)
        self.lateral_c5 = nn.Conv2d(c5, fpn_ch, 1, bias=False)

        self.fpn_out2 = ConvBNReLU(fpn_ch, fpn_ch, k=3, p=1)
        self.fpn_out3 = ConvBNReLU(fpn_ch, fpn_ch, k=3, p=1)
        self.fpn_out4 = ConvBNReLU(fpn_ch, fpn_ch, k=3, p=1)
        self.fpn_out5 = ConvBNReLU(fpn_ch, fpn_ch, k=3, p=1)

        self.fuse = ConvBNReLU(fpn_ch * 4, fpn_ch, k=3, p=1)
        self.classifier = nn.Conv2d(fpn_ch, num_classes, 1)

    def forward(self, c2, c3, c4, c5):
        p5_ppm = self.ppm(c5)
        p5 = self.lateral_c5(p5_ppm)

        p4 = self.lateral_c4(c4) + F.interpolate(
            p5, size=c4.shape[-2:], mode='bilinear', align_corners=False
        )
        p3 = self.lateral_c3(c3) + F.interpolate(
            p4, size=c3.shape[-2:], mode='bilinear', align_corners=False
        )
        p2 = self.lateral_c2(c2) + F.interpolate(
            p3, size=c2.shape[-2:], mode='bilinear', align_corners=False
        )

        p2 = self.fpn_out2(p2)
        p3 = self.fpn_out3(p3)
        p4 = self.fpn_out4(p4)
        p5 = self.fpn_out5(p5)

        p3_up = F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=p2.shape[-2:], mode='bilinear', align_corners=False)
        p5_up = F.interpolate(p5, size=p2.shape[-2:], mode='bilinear', align_corners=False)

        fused = self.fuse(torch.cat([p2, p3_up, p4_up, p5_up], dim=1))
        logits_small = self.classifier(fused)
        return logits_small, [p2, p3, p4, p5, fused]


class TeacherModel(nn.Module):
    """
    ConvNeXt-Tiny + UPerNet decoder.
    """
    BACKBONE_NAME = 'convnext_base'

    def __init__(self, num_classes=None, bb_channels=None, pretrained=True):
        super().__init__()
        num_classes = num_classes or config.NUM_CLASSES
        bb_channels = bb_channels or config.TEACHER_BB_CHANNELS

        self.backbone = timm.create_model(
            self.BACKBONE_NAME,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.head = UPerHead(
            in_channels=bb_channels,
            fpn_ch=256,
            num_classes=num_classes,
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        bb_feats = self.backbone(x)
        logits_small, dec_feats = self.head(
            bb_feats[0], bb_feats[1], bb_feats[2], bb_feats[3]
        )
        logits = F.interpolate(logits_small, size=(h, w), mode='bilinear', align_corners=False)
        return {
            'logits': logits,
            'bb_feats': bb_feats,
            'dec_feats': dec_feats,
        }


def build_teacher(freeze=True):
    model = TeacherModel().to(config.DEVICE)
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Teacher: {TeacherModel.BACKBONE_NAME} + UPerNet  "
          f"({n:.1f}M params, frozen={freeze})")
    return model
