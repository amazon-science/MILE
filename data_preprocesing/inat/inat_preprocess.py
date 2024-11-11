import os
import cv2
import numpy as np
from multiprocessing import Pool
from argparse import ArgumentParser
from typing import List, Tuple

SPLIT = 'train_val_images'
SUPERCLASSES = [
    'Actinopterygii', 'Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Chromista',
    'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia'
]

def process_image(image_path: str, target_path: str, location_prefix: str, md: str, suffix: str) -> None:
    img = cv2.imread(image_path)
    if img is not None:
        cv2.imwrite(f"{target_path}/{location_prefix}/{md}_{suffix}/{os.path.basename(image_path)}", img)
    else:
        print(f"Failed to read image: {image_path}")

def process_directory(args: Tuple[str, str, str]) -> None:
    md, data_path, target_path = args
    for sd in os.listdir(f"{data_path}/{SPLIT}/{md}/"):
        suffix = sd.replace(' ', '-')
        image_dir = f"{data_path}/{SPLIT}/{md}/{sd}"
        image_names = [x for x in os.listdir(image_dir) if x.endswith('.jpg')]

        os.makedirs(f"{target_path}/all/{md}_{suffix}", exist_ok=True)

        for idx, obj_img in enumerate(image_names):
            location_prefix = "query" if idx < 10 else "target"
            process_image(f"{image_dir}/{obj_img}", target_path, location_prefix, md, suffix)

def get_sorted_class_info(target_path: str) -> Tuple[List[str], List[int]]:
    class_dirs = os.listdir(f"{target_path}/all")
    class_count = [len(os.listdir(f"{target_path}/all/{c}")) for c in class_dirs]
    indices = np.argsort(class_count)[::-1]
    return [class_dirs[i] for i in indices], [class_count[i] for i in indices]

def print_superclass_info(class_dirs: List[str], class_count: List[int]) -> None:
    for sc in SUPERCLASSES:
        subset = [(item, count) for item, count in zip(class_dirs, class_count) if sc in item]
        for idx, (item, count) in enumerate(subset[:6]):
            print(f"{sc:<15} --- {item:<30} --- {count}")

def main(data_path: str, target_path: str) -> None:
    meta_dirs = os.listdir(f"{data_path}/{SPLIT}/")

    class_dirs, class_count = get_sorted_class_info(target_path)
    print("Top 10 indices:", np.argsort(class_count)[::-1][:10])

    print_superclass_info(class_dirs, class_count)

    with Pool() as pool:
        args = [(md, data_path, target_path) for md in meta_dirs]
        pool.map(process_directory, args)

if __name__ == '__main__':
    parser = ArgumentParser(description="Process iNaturalist dataset files")
    parser.add_argument("--data_path", type=str, default="./inat_raw_data/inference_by_class/images", help="Path to the data directory")
    parser.add_argument("--target_path", type=str, default="./inat_raw_data/inference_by_class", help="Path to the target directory")

    args = parser.parse_args()

    main(args.data_path, args.target_path)