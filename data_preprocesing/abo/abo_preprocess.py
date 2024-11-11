import argparse
import lzma
import pandas as pd
import os
import shutil
import logging
import tqdm
from typing import List

def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_fixed_splits(file_path: str) -> pd.DataFrame:
    """Load fixed splits from a compressed CSV file.
    
    Args:
        file_path: Path to the compressed CSV file.
        
    Returns:
        A DataFrame containing the loaded data.
    """
    try:
        with lzma.open(file_path, mode='rt') as f:
            return pd.read_csv(f)
    except Exception as e:
        logging.error(f"Error loading fixed splits: {e}")
        raise

def process_files(
    fixed_splits: pd.DataFrame, 
    abo_images_root: str, 
    abo_spins_root: str, 
    out_root: str, 
    splits: List[str] = ["test-query", "test-target"]
) -> None:
    """Process files based on the fixed splits and copy them to the output directory.
    
    Args:
        fixed_splits: DataFrame with fixed splits information.
        abo_images_root: Root directory for ABO images.
        abo_spins_root: Root directory for rendered ABO images.
        out_root: Root directory for output files.
        splits: List of split types to process.
    """
    os.makedirs(out_root, exist_ok=True)

    missed = 0
    found = 0
    for split in splits:
        split_out_dir = os.path.join(out_root, split.replace("-", "."))
        os.makedirs(split_out_dir, exist_ok=True)
        logging.info(f"Processing split: {split}")

        files = fixed_splits[fixed_splits["split_set"] == split]
        logging.info(f"{split} -> files: {len(files)}")
        for _, row in tqdm.tqdm(files.iterrows(), desc=split):
            if row.image_type == "render":
                info = row.path.split('/')
                infile = os.path.join(abo_spins_root, *info[1:])
            else:
                infile = os.path.join(abo_images_root, row.path)

            if not os.path.exists(infile):
                logging.warning(f"File not found: {infile}")
                missed += 1
                continue

            found += 1
            asin_out_dir = os.path.join(split_out_dir, str(row["class_id"]))
            os.makedirs(asin_out_dir, exist_ok=True)
            outfile = os.path.join(asin_out_dir, os.path.basename(row["path"]))
            shutil.copy(infile, outfile)

    logging.info(f"Missed: {missed}")
    logging.info(f"Found: {found}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ABO dataset files.")
    parser.add_argument("--input_csv", help="Path to the input CSV file", default="./abo_raw_data/abo-mvr.csv.xz")
    parser.add_argument("--abo_images_root", help="Root directory containing ABO images", default="./abo_raw_data/images/small/")
    parser.add_argument("--abo_spins_root", help="Root directory containing rendered ABO images", default="./abo_raw_data/images/small/")
    parser.add_argument("--out_root", help="Root directory for output files", default="./abo_raw_data/inference_by_class")
    args = parser.parse_args()

    setup_logging()

    try:
        fixed_splits = load_fixed_splits(args.input_csv)
        process_files(fixed_splits, args.abo_images_root, args.abo_spins_root, args.out_root)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
