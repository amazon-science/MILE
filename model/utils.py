import torch.nn as nn
from typing import Tuple
from dataclasses import dataclass

def trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    return trainable_params, all_params

@dataclass
class ViTConf:
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    num_classes: int = 0
    dynamic_img_size: bool = True

MODEL_CONFIGS = {
    "vit_b16_mae": ViTConf(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    "vit_b16_maws": ViTConf(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    "vit_l16_mae": ViTConf(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    "vit_l16_maws": ViTConf(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    "vit_h14_maws": ViTConf(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
}
