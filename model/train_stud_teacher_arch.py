# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import asdict
from model.model import CrossAttentionBlock
from dino.vision_transformer import VisionTransformer
from model.model import MILE
from model.utils import MODEL_CONFIGS, trainable_parameters
from peft import LoraConfig, get_peft_model

import torch
import dino.utils as utils
import dino.vision_transformer as vits
from dino.vision_transformer import DINOHead
from logger_config import logger

def build_teacher(args):
    args.arch = args.arch.replace("deit", "vit")
    if args.arch in vits.__dict__.keys():
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
    elif args.arch == "dinov2":
        teacher = torch.hub.load('facebookresearch/dinov2', args.model_name)
    elif args.arch == "maws":
        model_config = MODEL_CONFIGS[args.model_name]
        teacher = VisionTransformer(**asdict(model_config))
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    
    if args.distil_frozen_teacher:
        logger.info("LOADING TEACHER", args.frozen_teacher_checkpoint)
        msg = teacher.load_state_dict(torch.load(f"./base_models/{args.frozen_teacher_checkpoint}")["teacher"])
        assert str(msg) == "<All keys matched successfully>", msg
    
    return teacher

def apply_peft(student, teacher):
    config = LoraConfig(
        r=48,
        lora_alpha=16,
        target_modules=["qkv"],
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save = []
    )
    student = get_peft_model(student, config)
    teacher = get_peft_model(teacher, config)
    tp, ap = trainable_parameters(student)
    logger.info("AFTER LORA")
    logger.info(f"LoRA trainable params: {tp} || all params: {ap} || trainable%: {100 * tp / ap:.2f}")
    return student, teacher

def wrap_models(student, teacher, args):
    embed_dim = student.embed_dim
    
    if args.view == "multi-view":
        student_dual, teacher_dual = None, None
        if args.dual_condition:
            student_dual = CrossAttentionBlock(dim=embed_dim, proj_dim=2*embed_dim, explicit_residual=args.explicit_residual)
            teacher_dual = CrossAttentionBlock(dim=embed_dim, proj_dim=2*embed_dim, explicit_residual=args.explicit_residual)

        student_cross_wi_patch_e = args.cross_wi_patch_e if not args.distil_frozen_teacher else True
        teacher_cross_wi_patch_e = args.cross_wi_patch_e if not args.distil_frozen_teacher else False

        student = MILE(
            backbone=utils.MultiCropWrapper(teacher, None),
            latent_cross=CrossAttentionBlock(
                dim=embed_dim, 
                proj_dim=2 * embed_dim, 
                explicit_residual=args.explicit_residual, 
                num_heads=args.latent_cross_heads
            ),
            dual_latent_cross=student_dual,
            final_projection=DINOHead(
                embed_dim, 
                args.out_dim, 
                use_bn=args.use_bn_in_head, 
                norm_last_layer=args.norm_last_layer
            ),
            num_global_crops=2, 
            num_local_crops=args.local_crops_number, 
            cross_wi_patch_e=student_cross_wi_patch_e,
            gate_tanh=args.gate_tanh,
            dual_gate_tanh=args.dual_gate_tanh,
            bi_directional=args.bi_directional,
            union_latent_keys=args.union_latent_keys,
            backward_dual_latent_cross=args.backward_dual_latent_cross,
            ignore_cls_in_lxs=args.ignore_cls_in_lxs,
            cross_wi_registers=args.cross_wi_registers,
            arch=args.arch,
            dinov2_force_cls_in_lxs=args.dinov2_force_cls_in_lxs
        )
        
        
        teacher = MILE(
            backbone=utils.MultiCropWrapper(teacher, None),
            latent_cross=CrossAttentionBlock(
                dim=embed_dim, 
                proj_dim=2 * embed_dim, 
                explicit_residual=args.explicit_residual, 
                num_heads=args.latent_cross_heads
            ),
            dual_latent_cross=teacher_dual,
            final_projection=DINOHead(
                embed_dim, 
                args.out_dim, 
                use_bn=args.use_bn_in_head, 
                norm_last_layer=args.norm_last_layer
            ),
            num_global_crops=2, 
            num_local_crops=args.local_crops_number, 
            cross_wi_patch_e=teacher_cross_wi_patch_e,
            gate_tanh=args.gate_tanh,
            dual_gate_tanh=args.dual_gate_tanh,
            bi_directional=args.bi_directional,
            union_latent_keys=args.union_latent_keys,
            backward_dual_latent_cross=args.backward_dual_latent_cross,
            ignore_cls_in_lxs=args.ignore_cls_in_lxs,
            cross_wi_registers=args.cross_wi_registers,
            arch=args.arch,
            dinov2_force_cls_in_lxs=args.dinov2_force_cls_in_lxs
        )
        
    else:
        student = utils.MultiCropWrapper(student, DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ))
        teacher = utils.MultiCropWrapper(
            teacher,
            DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
        )
    
    return student, teacher
