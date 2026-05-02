import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


CLASS_NAMES = [f"class_{i}" for i in range(16)]


class ASPPConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPPPool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.AvgPool2d(kernel_size=32, stride=32),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        pooled = self.block(x)
        return F.interpolate(pooled, size=(32, 32), mode="bilinear", align_corners=False)


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, low_ch=48, high_ch=192, aspp_ch=256, low_proj_ch=48, num_classes=16):
        super().__init__()
        self.aspp_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(high_ch, aspp_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(aspp_ch),
                    nn.ReLU(inplace=True),
                ),
                ASPPConv(high_ch, aspp_ch, dilation=6),
                ASPPConv(high_ch, aspp_ch, dilation=12),
                ASPPConv(high_ch, aspp_ch, dilation=18),
                ASPPPool(high_ch, aspp_ch),
            ]
        )
        self.aspp_project = nn.Sequential(
            nn.Conv2d(aspp_ch * 5, aspp_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(aspp_ch),
            nn.ReLU(inplace=True),
        )
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch, low_proj_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_proj_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(aspp_ch + low_proj_ch, aspp_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(aspp_ch, aspp_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_ch),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(aspp_ch, num_classes, kernel_size=1)
        self.low_logits = nn.Conv2d(low_proj_ch, num_classes, kernel_size=1)
        self.kd_high_128 = nn.Conv2d(aspp_ch, 128, kernel_size=1, bias=False)
        self.kd_mid_128 = nn.Conv2d(aspp_ch, 128, kernel_size=1, bias=False)

    def forward(self, low_feat, high_feat):
        aspp_feats = [branch(high_feat) for branch in self.aspp_convs]
        aspp = self.aspp_project(torch.cat(aspp_feats, dim=1))
        aspp_up = F.interpolate(aspp, size=(128, 128), mode="bilinear", align_corners=False)

        low = self.low_proj(low_feat)
        fused = self.fuse(torch.cat([aspp_up, low], dim=1))
        logits = self.classifier(fused) + self.low_logits(low)
        return logits, aspp, fused


class Model2SegmentationModel(nn.Module):
    BACKBONE_NAME = "mobilenetv4_conv_large.e600_r384_in1k"

    def __init__(self, num_classes=16, bb_channels=None, is_export=False):
        super().__init__()
        self.is_export = is_export

        if bb_channels is None:
            bb_channels = [24, 48, 96, 192, 960]

        self.backbone = timm.create_model(
            self.BACKBONE_NAME,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        self.head = DeepLabV3PlusHead(
            low_ch=bb_channels[1],
            high_ch=bb_channels[3],
            aspp_ch=256,
            low_proj_ch=48,
            num_classes=num_classes,
        )

    def forward(self, x):
        bb_feats = self.backbone(x)
        logits_s, aspp_feat, fused_feat = self.head(bb_feats[1], bb_feats[3])
        logits = F.interpolate(logits_s, size=(512, 512), mode="bilinear", align_corners=False)

        if self.is_export:
            return logits

        return {
            "logits": logits,
            "bb_feats": bb_feats,
            "dec_feats": [aspp_feat, fused_feat],
        }


def build_model(num_classes=16, bb_channels=None, is_export=False, device="cpu"):
    model = Model2SegmentationModel(
        num_classes=num_classes,
        bb_channels=bb_channels,
        is_export=is_export,
    ).to(device)
    return model
