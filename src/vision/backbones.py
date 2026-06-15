"""Small pretrained image experts via timm, spanning the efficiency frontier.

The frontier is *drawn*, not hit at one point: a few genuinely tiny 2024-25
backbones at the cheap end, with MobileNetV5-300M (the Gemma 3n encoder) as the
heavy modern anchor to show diminishing returns. timm >= 1.0.16 is required for
the MobileNetV5 weights.

Each model is created head-less (num_classes=0) so it yields an embedding; a
light classifier head is attached for the malignancy logit. We extract BOTH the
OOF probability and the pooled embedding for stacking into the GBDT.
"""
from __future__ import annotations

import torch.nn as nn

# name -> timm model id. Cheap end first, heavy anchor last.
#
# All ids confirmed loadable on timm 1.0.27, natural-image pretrain only (no
# skin-cancer labels), so the no-external-*data* claim survives. The set spans
# the cost frontier: tiny mobile/hybrid CNNs at the cheap end, the proven
# small/modern encoders from the top ISIC-2024 open-source solutions in the
# middle, and MobileNetV5-300M (the Gemma 3n encoder) as the heavy anchor.
#
# Provenance of the added encoders (top ISIC-2024 solutions, image-only recipe):
#   convnextv2_nano  3rd-place primary; .fcmae = MAE self-supervised init then
#                    in22k->in1k fine-tune (still natural-image pretrain, no skin).
#   vit_tiny         3rd place, reported CV 0.161 (augreg in21k -> in1k).
#   vit_small        sibling ViT anchor (augreg in21k -> in1k).
#   eva02_small      heavier ViT anchor, 336px native (MIM in22k -> in1k).
#   swinv2_tiny      hierarchical-transformer anchor, 256px native.
#
# NOTE on input size: the transformer-family ids (vit_*, eva02_*, swinv2_*) carry
# a NATIVE training resolution (224 / 336 / 256). ImageExpert passes the runtime
# ``img_size`` into ``timm.create_model`` so timm interpolates the position
# embeddings to our 128px efficiency target. CNNs ignore/reject ``img_size`` (they
# are fully convolutional); we handle that with a graceful fallback.
FRONTIER = {
    # --- cheap end: tiny mobile / hybrid backbones ---
    "mnv4_small":      "mobilenetv4_conv_small.e2400_r224_in1k",
    "starnet_s1":      "starnet_s1.in1k",
    "ghostnetv3":      "ghostnetv3_100.in1k",          # loads on timm 1.0.27
    "effvit_b0":       "efficientvit_b0.r224_in1k",
    "effnetv2_b0":     "tf_efficientnetv2_b0.in1k",
    "fastvit_t8":      "fastvit_t8.apple_in1k",
    # --- proven small/modern encoders from top ISIC-2024 solutions ---
    "convnextv2_nano": "convnextv2_nano.fcmae_ft_in22k_in1k",      # 3rd-place primary
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",      # bigger ConvNeXt-V2 anchor
    "vit_tiny":        "vit_tiny_patch16_224.augreg_in21k_ft_in1k",  # 3rd place CV 0.161
    "vit_small":       "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "eva02_small":     "eva02_small_patch14_336.mim_in22k_ft_in1k",  # 336px native
    "swinv2_tiny":     "swinv2_tiny_window8_256.ms_in1k",            # 256px native
    # --- heavy anchor ---
    "mnv5_300m":       "mobilenetv5_300m.gemma3n",      # needs timm>=1.0.16; sane on 1.0.27
}


class ImageExpert(nn.Module):
    def __init__(self, timm_id: str, pretrained: bool = True, drop: float = 0.2,
                 img_size: int = 128):
        super().__init__()
        import timm
        import torch

        # Transformer-family backbones need ``img_size`` so timm interpolates the
        # position embeddings from their native resolution to ours. Fully
        # convolutional backbones reject the kwarg -> fall back without it.
        try:
            self.backbone = timm.create_model(
                timm_id, pretrained=pretrained, num_classes=0,
                global_pool="avg", img_size=img_size,
            )
        except TypeError:
            self.backbone = timm.create_model(
                timm_id, pretrained=pretrained, num_classes=0, global_pool="avg",
            )

        # timm's reported `num_features` can disagree with the actual pooled
        # output for some backbones (e.g. MobileNetV4's conv_head expands
        # 960->1280 while num_features still reports 960; GhostNetV3 likewise),
        # so probe the true embedding dim with a dry forward at the RUNTIME
        # img_size. Robust across every FRONTIER backbone, including the
        # transformers whose pos-embeds were just interpolated to img_size.
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            self.embed_dim = int(
                self.backbone(torch.zeros(1, 3, img_size, img_size)).shape[1]
            )
        self.backbone.train(was_training)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(self.embed_dim, 1))

    def forward(self, x, return_embedding: bool = False):
        z = self.backbone(x)
        logit = self.head(z).squeeze(-1)
        return (logit, z) if return_embedding else logit


def build(name: str, **kw) -> ImageExpert:
    if name not in FRONTIER:
        raise KeyError(f"unknown backbone {name!r}; choose from {list(FRONTIER)}")
    return ImageExpert(FRONTIER[name], **kw)
