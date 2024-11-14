import os
from typing import List, Optional, Dict, Any

import datasets
from datasets import Dataset, DatasetDict

from configs.local_config import LOCAL_S3_MAP
from utils.s3_utils import get_processed_cache_name
from logger_config import logger

class TransformDataset(Dataset):
    def __init__(self, subset: Dataset, transform: callable):
        self.subset = subset
        self.transform = transform

    def get_one_item(self, index: int) -> Dict[str, Any]:
        return self.transform(self.subset[index])

    def __getitem__(self, index: int | List[int]) -> Dict[str, Any] | Dict[str, List[Any]]:
        if not isinstance(index, list):
            return self.get_one_item(index)

        items = [self.get_one_item(i) for i in index]
        return {k: [item[k] for item in items] for k in items[0].keys()}

    def set_format(self, format: str):
        self.subset.set_format(format)

    def __len__(self) -> int:
        return len(self.subset)

def load_raw_datasets(args: Any) -> DatasetDict:
    data_cache_dir = os.path.join(args.data_cache, args.dataset_name, "main")
    local_data_cache_dir = os.path.join(LOCAL_S3_MAP, data_cache_dir)

    logger.info.info(f"Data cache dir: {data_cache_dir}")
    logger.info.info(f"Local data cache dir: {local_data_cache_dir}")

    if os.path.exists(local_data_cache_dir):
        logger.info(f"Loading serialized dataset from {local_data_cache_dir}")
        datasets_map = datasets.load_from_disk(local_data_cache_dir)
    else:
        logger.info("Loading raw dataset")
        datasets_map = datasets.load_dataset("./data/AboDataset", args.dataset_name)
        logger.info(f"Saving to local: {local_data_cache_dir}")
        datasets_map.save_to_disk(local_data_cache_dir)
        logger.info(f"Saving to S3: s3://{args.s3_bucket}/{data_cache_dir}")
        datasets_map.save_to_disk(os.path.join("s3://", args.s3_bucket, data_cache_dir))

    logger.info(f"Data schema: {datasets_map}")
    return datasets_map

def load_datasets(
    args: Any,
    transform: callable,
    remove_columns: List[str] = [],
    interest_splits: List[str] = ["train"],
    data_type: Optional[str] = None
) -> DatasetDict:
    processed_cache_name = get_processed_cache_name(args)
    if data_type is not None:
        processed_cache_name = f"{processed_cache_name}_{data_type}"
    
    data_cache_dir = os.path.join(args.data_cache, args.dataset_name, "main")
    local_data_cache_dir = os.path.join(LOCAL_S3_MAP, data_cache_dir)
    processed_cache_dir = os.path.join(args.data_cache, args.dataset_name, processed_cache_name)
    local_processed_cache_dir = os.path.join(LOCAL_S3_MAP, processed_cache_dir)

    logger.info(f"Data cache dir: {data_cache_dir}")
    logger.info(f"Local data cache dir: {local_data_cache_dir}")
    logger.info(f"Processed cache dir: {processed_cache_dir}")
    logger.info(f"Local processed cache dir: {local_processed_cache_dir}")

    if os.path.exists(local_processed_cache_dir):
        logger.info(f"Loading processed data from {local_processed_cache_dir}")
        return datasets.load_from_disk(local_processed_cache_dir)

    if os.path.exists(local_data_cache_dir):
        logger.info(f"Loading serialized dataset from {local_data_cache_dir}")
        datasets_map = datasets.load_from_disk(local_data_cache_dir)
    else:
        logger.info("Loading raw dataset")
        datasets_map = datasets.load_dataset("./data/AboDataset", args.dataset_name)

    logger.info(f"Data schema: {datasets_map}")

    if args.sample:
        datasets_map = DatasetDict({
            mode: dataset.select(range(0, args.batch_size_per_gpu * 3))
            for mode, dataset in datasets_map.items()
        })

    logger.info(f"Applying processing for view: {args.view}")
    datasets_map = datasets_map.map(
        transform,
        batched=False,
        batch_size=args.batch_size_per_gpu,
        remove_columns=remove_columns,
    )

    # Save processed dataset
    datasets_map.save_to_disk(local_processed_cache_dir)
    datasets_map.save_to_disk(os.path.join("s3://", args.s3_bucket, processed_cache_dir))

    logger.info(f"Processed datasets: {datasets_map}")
    return datasets_map