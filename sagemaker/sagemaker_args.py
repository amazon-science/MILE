# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import dino.utils as utils
import argparse


def get_sagemaker_args():
    parser = argparse.ArgumentParser("SGM", add_help=False)

    # SageMaker training job args
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--local_run", action="store_true", default=False)
    parser.add_argument("--s3_bucket", type=str, default="cosine-irl")
    parser.add_argument("--data_cache", type=str, default="data/abo/cache")
    parser.add_argument("--dataset_name", type=str, default="Abo_product_type")

    parser.add_argument("--sample", action="store_true", default=False)
    parser.add_argument("--find_batch", action="store_true", default=False)

    parser.add_argument("--ckpt_root", type=str, default="/shared/projects/iblrp/runs/")
    parser.add_argument("--ckpt_name", type=str, default=None)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--base_model_name", type=str, default="dinov2_vitl14_reg")

    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--cls_type", type=str, default="knn")

    parser.add_argument("--test_source", type=str, default="test")
    parser.add_argument("--target_source", type=str, default="train")

    parser.add_argument("--knn_index", type=str, default=None)
    parser.add_argument("--knn_votes", type=int, default=1)
    parser.add_argument("--top_interest_labels", type=int, default=None)
    parser.add_argument("--k_max", type=int, default=10)

    parser.add_argument("--knn_sample_train", type=float, default=1)
    parser.add_argument("--acc_steps", type=int, default=1)
    parser.add_argument("--label_map", type=str, default=None)
    parser.add_argument("--inference_res", type=int, default=224)
    parser.add_argument("--output_type", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--distributed_mode", type=utils.bool_flag, default=False)

    # multi view
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=4,
        help="""Number of samples per class if view=multi-view""",
    )

    return parser
