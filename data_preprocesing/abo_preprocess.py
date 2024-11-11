import os
import shutil
import lzma
import argparse
import logging
import pandas as pd

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_fixed_splits(file_path):
    try:
        with lzma.open(file_path, mode='rt') as f:
            return pd.read_csv(f)
    except Exception as e:
        logging.error(f"Error loading fixed splits: {e}")
        raise

def process_files(fixed_splits, in_class_root, out_root):
    """Process files based on the fixed splits and copy them to the output directory."""
    missed = 0
    found = 0

    for split in ["test-query", "test-target"]:
        out_class_root = os.path.join(out_root, split)
        os.makedirs(out_class_root, exist_ok=True)

        files = fixed_splits[fixed_splits["split_set"] == split]
        logging.info(f"{split}: {len(files)} files")

        for _, row in files.iterrows():
            relative_path = os.path.join(str(row["class_id"]), os.path.basename(row["path"]))
            infile = os.path.join(in_class_root, relative_path)
            if not os.path.exists(infile):
                logging.warning(f"Missing file: {infile}")
                missed += 1
                continue
            found += 1
            cls_path = os.path.join(out_class_root, str(row["class_id"]))
            os.makedirs(cls_path, exist_ok=True)
            outfile = os.path.join(out_class_root, relative_path)
            shutil.copy(infile, outfile)

    logging.info(f"Missed: {missed}")
    logging.info(f"Found: {found}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ABO dataset files.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file")
    parser.add_argument("--in_class_root", required=True, help="Root directory of input class files")
    parser.add_argument("--out_root", required=True, help="Root directory for output files")
    args = parser.parse_args()

    setup_logging()

    try:
        fixed_splits = load_fixed_splits(args.input_csv)
        process_files(fixed_splits, args.in_class_root, args.out_root)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise