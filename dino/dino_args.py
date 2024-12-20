# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Original Copyright 2021 Facebook, Inc. and its affiliates. Licensed under the Apache License, Version 2.0
# Modifications Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import argparse
import dino.utils as utils

def get_dino_args():
    parser = argparse.ArgumentParser('MultiView', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small', 'dinov2', 'maws'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--gradient_acc_steps', default=1, type=int,
        help='Gradient accumulation steps.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="./out", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
        
    parser.add_argument('--freeze_pre_trained', type=utils.bool_flag, default=False)
    parser.add_argument('--explicit_residual', type=utils.bool_flag, default=False)
    parser.add_argument("--dual_condition", type=utils.bool_flag, default=False)
    parser.add_argument("--bi_directional", type=utils.bool_flag, default=False)
    parser.add_argument("--union_latent_keys", type=utils.bool_flag, default=False)
    parser.add_argument("--backward_dual_latent_cross", type=utils.bool_flag, default=False)
    parser.add_argument("--subset_classes", default=None, type=int, help="subset classes")
    parser.add_argument("--no_pad", type=utils.bool_flag, default=False)
    parser.add_argument("--deterministic", type=utils.bool_flag, default=False)
    parser.add_argument("--latent_cross_heads", type=int, default=1)

    parser.add_argument("--duplicate_samples", type=int, default=1)
    parser.add_argument("--reverse_duplicate_samples", type=utils.bool_flag, default=True)
    parser.add_argument('--model_source', default='teacher', type=str)

    parser.add_argument("--cross_wi_registers", type=utils.bool_flag, default=False)
    parser.add_argument("--fold_cls_registers", type=utils.bool_flag, default=False)
    # 2024-11-22: Amazon addition.
    parser.add_argument('--inference_noise_index', type=int, default=None)
    parser.add_argument('--stitching', type=utils.bool_flag, default=False)
    parser.add_argument("--input_order", type=int, default=-1)
    # 2024-11-22: End of Amazon addition.
    return parser

# 2024-11-22: Amazon addition.
def get_mile_args():
    parser = argparse.ArgumentParser('MILE', add_help=False)

    # MILE-specific parameters
    parser.add_argument('--base_name', default='ty2-86.23', type=str, help='Base name for the experiment')
    parser.add_argument('--view', default='multi-view', type=str, help='View type')
    parser.add_argument('--cross_wi_patch_e', default=False, type=utils.bool_flag, help='Cross with patch embedding')
    parser.add_argument('--ignore_cls_in_lxs', default=False, type=utils.bool_flag, help='Ignore CLS token in LXS')
    parser.add_argument('--dinov2_force_cls_in_lxs', default=False, type=utils.bool_flag, help='Force CLS token in LXS for DINOv2')
    parser.add_argument('--gate_tanh', default=True, type=utils.bool_flag, help='Use tanh in gate')
    parser.add_argument('--peft', default=True, type=utils.bool_flag, help='Use PEFT')
    parser.add_argument('--distil_frozen_teacher', default=False, type=utils.bool_flag, help='Distil from frozen teacher')
    parser.add_argument('--dual_gate_tanh', default=False, type=utils.bool_flag, help='Deprecated // independent tanh gating')

    return parser
# 2024-11-22: End of Amazon addition.