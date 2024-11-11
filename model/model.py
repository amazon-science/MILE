import logging
from dataclasses import asdict, dataclass
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from timm.models.vision_transformer import VisionTransformer
from peft import LoraConfig, get_peft_model

import dino.vision_transformer as vits
import dino.utils as utils
from dino.utils import MILE, CrossAttentionBlock
from dino.vision_transformer import DINOHead

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def get_backbone(type: str, model_name: str, patch_size: int, num_classes: int) -> nn.Module:
    logging.info(f"Selecting backbone for {model_name}, type: {type}")
    
    if type.startswith("dinov2"):
        options = ["dinov2-vits14", "dinov2-vitb14", "dinov2-vitl14-reg", "dinov2-vitl14"]
        for option in options:
            if option in model_name:
                logging.info(f"Loading base {option}")
                return torch.hub.load("facebookresearch/dinov2", option.replace("-", "_"))
        raise ValueError(f"{model_name} missing from {options}")
    
    elif "maws" in type or "mae" in type:
        for option, config in MODEL_CONFIGS.items():
            model_key = option.replace("_", "-")
            if model_key in model_name:
                model = VisionTransformer(**asdict(config))
                logging.info(f"Loading {option}")
                state_dict = torch.load(f"./dino_distributed_training/dino_distributed/base_models/{option}.pt")
                msg = model.load_state_dict(state_dict)
                if str(msg) != "<All keys matched successfully>":
                    raise ValueError(f"Error loading state dict: {msg}")
                return model
        raise ValueError(f"{model_name} missing from {MODEL_CONFIGS.keys()}")
    
    else:  # DinoV1
        return vits.__dict__[type](patch_size=patch_size, num_classes=num_classes)

def trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    return trainable_params, all_params

def init_model(args: Any, weight_path: Optional[str], num_classes: int, device: str, peft: bool = False) -> nn.Module:
    model = get_backbone(args.arch, args.base_model_name, args.patch_size, num_classes)
    
    if args.view == "single":
        if weight_path:
            state_dict = torch.load(weight_path, map_location="cpu")[args.model_source]
            state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        
        if peft:
            config = LoraConfig(
                r=48, lora_alpha=16, target_modules=["qkv"], lora_dropout=0.1,
                bias="lora_only", modules_to_save=[]
            )
            model = get_peft_model(model, config)
            tp, ap = trainable_parameters(model)
            logging.info(f"LoRA trainable params: {tp} || all params: {ap} || trainable%: {100 * tp / ap:.2f}")
    
    elif args.view == "multi-view":
        if peft:
            config = LoraConfig(
                r=48, lora_alpha=16, target_modules=["qkv"], lora_dropout=0.1,
                bias="lora_only", modules_to_save=[]
            )
            model = get_peft_model(model, config)
            tp, ap = trainable_parameters(model)
            logging.info(f"LoRA trainable params: {tp} || all params: {ap} || trainable%: {100 * tp / ap:.2f}")
        
        embed_dim = model.embed_dim
        dual_latent_cross = CrossAttentionBlock(dim=embed_dim, proj_dim=2*embed_dim, explicit_residual=args.explicit_residual) if args.dual_condition else None
        
        model = MILE(
            backbone=utils.MultiCropWrapper(model, None),
            latent_cross=CrossAttentionBlock(
                dim=embed_dim, proj_dim=2*embed_dim, explicit_residual=args.explicit_residual,
                num_heads=args.latent_cross_heads
            ),
            dual_latent_cross=dual_latent_cross,
            final_projection=DINOHead(
                embed_dim, args.out_dim, use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer
            ),
            num_global_crops=2,
            num_local_crops=args.local_crops_number,
            bi_directional=args.bi_directional,
            union_latent_keys=args.union_latent_keys,
            backward_dual_latent_cross=args.backward_dual_latent_cross,
            cross_wi_patch_e=args.cross_wi_patch_e,
            ignore_cls_in_lxs=args.ignore_cls_in_lxs,
            cross_wi_registers=args.cross_wi_registers,
            arch=args.arch,
            dinov2_force_cls_in_lxs=args.dinov2_force_cls_in_lxs,
        )
        
        if weight_path:
            logging.info(f"Loading model from {weight_path}, source: {args.model_source}")
            state_dict = torch.load(weight_path, map_location="cpu")[args.model_source]
            if args.model_source == "student":
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
    
    else:
        raise ValueError(f"Unknown view: {args.view}")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    return model

def process_embeddings(args: Any, model: nn.Module, dataset: Any, output_type: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    # batch size set to 1 since 1 batch contains the entire image-set to be processed by MILE
    loader = DataLoader(
        dataset, shuffle=False, batch_size=1,
        num_workers=1, pin_memory=True, drop_last=False
    )
    results: List[Tuple[torch.Tensor, torch.Tensor]] = []
    
    for batch_index, batch in tqdm.tqdm(enumerate(loader), desc="Processing embeddings"):
        images, labels = batch
        if device == "cuda":
            images = [view.cuda() for view in images]
        
        if args.view == "single":
            cls_tokens = [model(image) for image in images]
            for cls in cls_tokens:
                results.append((cls, labels))
        elif args.view.startswith("multi-view"):
            cls_tokens = model(images, output_type=output_type)
            cls_tokens = cls_tokens.to("cpu")
            if args.cross_wi_registers:
                cls_tokens = cls_tokens[:, 0, :]
            labels = labels.to("cpu")
            results.append((cls_tokens, labels))
        else:
            raise ValueError(f"Unknown view: {args.view}")

    X = torch.cat([e[0] for e in results], dim=0)
    Y = torch.cat([e[1] for e in results], dim=0)
    return {"X": X, "Y": Y}
