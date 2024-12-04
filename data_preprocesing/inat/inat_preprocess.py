# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import cv2
import random
from argparse import ArgumentParser
from typing import Tuple
import glob
import shutil
import tqdm

SPLIT = 'train_val_images'
SUPERCLASSES = [
    'Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Chromista',
    'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia'
]

def process_directory(args: Tuple[str, str, str, int]) -> None:
    md, data_path, target_path, num_sub_classes = args
    classes = os.listdir(f"{data_path}/{SPLIT}/{md}/")
    random.shuffle(classes)
    if len(classes) > num_sub_classes:
        classes = classes[:num_sub_classes]
    for sd in classes:
        suffix = sd.replace(' ', '-')
        image_dir = f"{data_path}/{SPLIT}/{md}/{sd}"
        image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
        th_index = len(image_files) // 2
        for idx, img_file in enumerate(image_files):
            location_prefix = "query" if idx <= th_index else "target"
            outdir = f"{target_path}/{location_prefix}/{md}_{suffix}/"
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, os.path.basename(img_file))
            shutil.copy(img_file, outfile)

def main(data_path: str, target_path: str, num_sub_classes: int) -> None:
    meta_dirs = os.listdir(f"{data_path}/{SPLIT}/")
    for md in tqdm.tqdm(meta_dirs):
        args = (md, data_path, target_path, num_sub_classes)
        process_directory(args)

if __name__ == '__main__':
    parser = ArgumentParser(description="Process iNaturalist dataset files")
    parser.add_argument("--data_path", type=str, default="./inat_raw_data/", help="Path to the data directory")
    parser.add_argument("--target_path", type=str, default="./inat_raw_data/inference_by_class", help="Path to the target directory")
    parser.add_argument("--num_sub_classes", type=int, default=100, help="Number of classes per species")
    
    args = parser.parse_args()
    main(args.data_path, args.target_path, args.num_sub_classes)