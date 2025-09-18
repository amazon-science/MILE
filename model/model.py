import logging
from dataclasses import asdict, dataclass
from typing import Dict, Any, Optional, List, Tuple

from dino.utils import trunc_normal_
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from timm.models.vision_transformer import VisionTransformer
from peft import LoraConfig, get_peft_model
import numpy as np

import dino.vision_transformer as vits
import dino.utils as utils
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

def get_backbone(type: str, base_model_name: str) -> nn.Module:
    logging.info(f"Selecting backbone for {base_model_name}, type: {type}")
    
    if type.startswith("dinov2"):
        return torch.hub.load("facebookresearch/dinov2", base_model_name)
    elif "maws" in type or "mae" in type:
        return torch.hub.load("facebookresearch/maws", base_model_name)
    else:
        logging.error(f"Unknown arch type {type}")
        return None

def trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    return trainable_params, all_params

def get_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class MILE(nn.Module):
    def __init__(
        self,
        backbone,
        latent_cross,
        dual_latent_cross,
        num_global_crops,
        num_local_crops,
        final_projection=None,
        gate_tanh=False,
        dual_gate_tanh=False,
        bi_directional=False,
        union_latent_keys=False,
        backward_dual_latent_cross=False,
        cross_wi_patch_e=False,
        ignore_cls_in_lxs=False,
        cross_wi_registers=False,
        arch=None,
        dinov2_force_cls_in_lxs=False
    ):
        super(MILE, self).__init__()
        
        # Backbone and processing components
        self.backbone = backbone  # ViT backbone
        self.latent_cross = latent_cross  # Aggregator module
        self.final_projection = final_projection  # DINO head
        self.dual_latent_cross = dual_latent_cross  # Dual condition

        # Crop configuration
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.ncrops = num_local_crops + num_global_crops

        # Gate parameters
        self.gate_alpha = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.dual_gate_alpha = nn.Parameter(torch.tensor(0.), requires_grad=False)

        # Flags and configurations
        self.bi_directional = bi_directional
        self.union_latent_keys = union_latent_keys
        self.backward_dual_latent_cross = backward_dual_latent_cross
        self.cross_wi_patch_e = cross_wi_patch_e
        self.ignore_cls_in_lxs = ignore_cls_in_lxs
        self.dinov2_force_cls_in_lxs = dinov2_force_cls_in_lxs
        self.arch = arch

        if gate_tanh:
            self.gate_alpha.requires_grad = True

        if dual_gate_tanh:
            self.dual_gate_alpha.requires_grad = True

        self.cross_wi_registers = cross_wi_registers

    def forward(self, x, output_type=None):
        assert(isinstance(x, list))
        #assert len(x) > 1, "expecting more than a single view" # more than 1 view
        latent = None
        outputs = []
        latents = []

        for view in x:
            if self.cross_wi_patch_e:
                output = [self.backbone.backbone.forward_features(crop_view.unsqueeze(0)) for crop_view in view]
                if self.ignore_cls_in_lxs:
                    output = [crop_output[:, 1:, :] for crop_output in output] # [BSxSeqxHidden] # drop cls 
            elif self.cross_wi_registers:
                output = self.backbone.backbone.forward_features(view)
                output = torch.cat([output["x_norm_clstoken"].unsqueeze(1), output["x_norm_regtokens"]], dim=1)
                output = [output]
            else:
                output = self.backbone(view)
                output = list(output.chunk(self.ncrops))

            if self.dual_latent_cross and latent:
                assert self.cross_wi_patch_e == False, "Dual condition not implemented for latent cross with embeddings"
                for crop_idx in range(len(latent)):
                    dual_alpha = nn.Tanh()(self.dual_gate_alpha)
                    output[crop_idx] = (1 - dual_alpha) * self.dual_latent_cross(Q=output[crop_idx], KV=latent[crop_idx], X=output[crop_idx]) + dual_alpha * output[crop_idx]

            outputs.append(output)

            if latent is not None:
                assert(len(latent) == len(output))

                if self.arch == "dinov2" and self.cross_wi_patch_e: ### TODO: UGLY
                    if self.dinov2_force_cls_in_lxs: ## UGLY , for dinov2 ignore_cls_in_lxs is not used
                        output = [torch.cat([crop["x_norm_clstoken"].unsqueeze(1), crop["x_norm_patchtokens"]], dim=1) for crop in output]
                    else:
                        output = [crop["x_norm_patchtokens"] for crop in output]

                for crop_idx in range(len(latent)):
                    alpha = nn.Tanh()(self.gate_alpha)
                    kv = output[crop_idx] # either full embeddings sequence or cls
                    q = latent[crop_idx]
                    if self.cross_wi_patch_e: # 2D attention
                        q = q.unsqueeze(1)
                    latent[crop_idx] = (1-alpha) * self.latent_cross(Q=q, KV=kv, X=latent[crop_idx]) + alpha * latent[crop_idx]
            else:
                latent = output
                if self.cross_wi_patch_e:
                    for crop_idx in range(len(latent)):
                        if self.arch == "dinov2": ### TODO: UGLY
                            latent[crop_idx] = self.backbone.backbone.head(latent[crop_idx]["x_norm_clstoken"])
                        else:
                            latent[crop_idx] = self.backbone.backbone.forward_head(latent[crop_idx])

            latents.append(latent)

        if self.bi_directional:
            for idx in range(len(outputs)-2, -1, -1):
                output = outputs[idx]
                assert(len(latent) == len(output))

                if self.backward_dual_latent_cross:
                    for crop_idx in range(len(latent)):
                        dual_alpha = nn.Tanh()(self.dual_gate_alpha)
                        output[crop_idx] = (1 - dual_alpha) * self.dual_latent_cross(Q=output[crop_idx], KV=latent[crop_idx], X=output[crop_idx]) + dual_alpha * output[crop_idx]

                for crop_idx in range(len(latent)):
                    alpha = nn.Tanh()(self.gate_alpha)
                    if self.union_latent_keys:
                        latent_KV = torch.cat([output[crop_idx], latents[idx][crop_idx]])
                        latent[crop_idx] = (1-alpha) * self.latent_cross(Q=latent[crop_idx], KV=latent_KV, X=latent[crop_idx]) + alpha * latent[crop_idx]
                    else:
                        latent[crop_idx] = (1-alpha) * self.latent_cross(Q=latent[crop_idx], KV=output[crop_idx], X=latent[crop_idx]) + alpha * latent[crop_idx]

        output = torch.cat(latent)
        if output_type == "latent":
            return output
        elif output_type:
            assert False, f"uknown {output_type}"

        if self.final_projection is not None:
            output = self.final_projection(output)
        return output


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, proj_dim, num_heads=1, attn_dropout=0, explicit_residual=False):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_post_attn = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(
                        embed_dim=dim,
                        num_heads=num_heads,
                        dropout=attn_dropout,
                        batch_first=True)

        self.mlpblock = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, dim),
        )
        self.apply(self._init_weights)

        print("explicit residual", explicit_residual)
        print("num heads", num_heads)
        self.explicit_residual = explicit_residual

    def forward(self, Q, KV, X):
        Q = self.norm_q(Q)
        KV = self.norm_kv(KV)
        X_attn = self.mha(
            query=Q,
            key=KV,
            value=KV,
            need_weights=False)[0].squeeze() + X
        if self.explicit_residual:
            X_res = X
        else:
            X_res = X_attn
        X = self.mlpblock(self.norm_post_attn(X_attn)) + X_res
        return X

    def _init_weights(self, m):
        # print("called init weights for", m)
        from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
        if isinstance(m, nn.Linear) or isinstance(m, NonDynamicallyQuantizableLinear):
            print("\tinit linear", m)
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)




def init_model(args: Any, weight_path: Optional[str], num_classes: int, device: str, peft: bool = False) -> nn.Module:
    model = get_backbone(args.arch, args.base_model_name)
    
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
