import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import config

# ============================================================
# HAILO-8 OPTIMIZED STUDENT MODEL
# Changes: 
# 1. Fixed interpolation sizes (Static Graph)
# 2. Hardsigmoid -> Sigmoid (Native Hailo APU support)
# 3. ReLU6 -> ReLU (Quantization stability)
# 4. Clean Tensor Output (No dictionary for ONNX export)
# ============================================================

class LRASPPHead(nn.Module):
    def __init__(self, low_ch, high_ch, inter_ch=128, num_classes=7):
        super().__init__()
        self.high_conv = nn.Sequential(
            nn.Conv2d(high_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True), # ReLU6 -> ReLU for Hailo compatibility
        )
        self.gap_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_ch, inter_ch, 1, bias=False),
            nn.Sigmoid(), # Hardsigmoid -> Sigmoid (Cleaner ONNX node)
        )
        self.low_conv   = nn.Conv2d(low_ch, num_classes, 1, bias=False)
        self.classifier = nn.Conv2d(inter_ch, num_classes, 1, bias=False)

    def forward(self, low_feat, high_feat):
        high_out = self.high_conv(high_feat)
        gate     = self.gap_gate(high_feat)
        gated    = high_out * gate
        
        # FIXED SIZE: Use static tuple for Hailo. 
        # For IDD, this is usually 1/4 of input resolution (e.g., 128x128 for 512x512 input)
        gated_up = F.interpolate(gated, size=(128, 128), mode='bilinear', align_corners=False)
        
        high_logits = self.classifier(gated_up)
        low_logits  = self.low_conv(low_feat)
        return high_logits + low_logits

class StudentModel(nn.Module):
    BACKBONE_NAME = 'mobilenetv4_conv_small.e2400_r224_in1k'

    def __init__(self, num_classes=None, bb_channels=None, is_export=False):
        super().__init__()
        self.is_export = is_export # Toggle for Training vs. ONNX Export
        num_classes = num_classes or config.NUM_CLASSES
        bb_channels = bb_channels or config.STUDENT_BB_CHANNELS

        self.backbone = timm.create_model(
            self.BACKBONE_NAME,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.head = LRASPPHead(
            low_ch     = bb_channels[1], 
            high_ch    = bb_channels[3], 
            inter_ch   = 128,
            num_classes= num_classes,
        )

    def forward(self, x):
        bb_feats = self.backbone(x)
        logits_s = self.head(bb_feats[1], bb_feats[3])
        
        # FINAL UPSCALE: Fixed to your target input size (e.g., 512x512)
        logits = F.interpolate(logits_s, size=(512, 512), mode='bilinear', align_corners=False)

        # IMPORTANT: During export, we return ONLY the tensor.
        # During training, we return the dict for the KD loss logic.
        if self.is_export:
            return logits
            
        return {
            'logits'   : logits,
            'bb_feats' : bb_feats,
            'dec_feats': [],
        }

def build_student(is_export=False):
    model = StudentModel(is_export=is_export).to(config.DEVICE)
    return model