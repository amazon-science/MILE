# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import boto3
import os

from logger_config import logger

def get_processed_cache_name(args):
    name = "RND_"
    if args.sample:
        name += "sample_"

    fields = [("data", args.dataset_name), ("view", args.view)]
    name = name + "_".join(["=".join(field) for field in fields])

    return name


def get_job_name(args):
    name = args.name
    name = name + "_" + get_processed_cache_name(args)
    name = name + "_" + get_processed_cache_name(args)
    fields = [
        ("arch", args.arch),
        ("P", args.patch_size),
        ("BS", args.batch_size_per_gpu * args.acc_steps),
        ("E", args.epochs),
        ("Ew", args.warmup_epochs),
        ("lr", args.lr),
        ("Frz", args.freeze_last_layer),
        ("NormLL", args.norm_last_layer),
        ("F16", args.use_fp16),
        ("H", args.out_dim),
    ]

    fields = [[str(e) for e in entry] for entry in fields]
    name = name + "_" + "_".join(["=".join(field) for field in fields])
    return name


def s3_folder_exists(bucket: str, path: str) -> bool:
    """
    Folder should exists.
    Folder could be empty.
    """
    s3 = boto3.client("s3")
    path = path.rstrip("/")
    resp = s3.list_objects(Bucket=bucket, Prefix=path, Delimiter="/", MaxKeys=1)
    return "CommonPrefixes" in resp


def download_bucket(input_path, output_path, bucket_name):
    os.makedirs(output_path, exist_ok=True)
    bucket = boto3.resource("s3").Bucket(bucket_name)
    data_files = bucket.objects.filter(Prefix=input_path)
    logger.info(f"downloading: {input_path} -> {output_path}")
    for file in data_files:
        key = file.key
        path = key.replace(input_path, "")
        target_dir, fname = os.path.split(path)
        outdir = os.path.join(output_path, target_dir)
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, fname)
        logger.info("downloading", key, outfile)
        bucket.download_file(key, outfile)
    return output_path
