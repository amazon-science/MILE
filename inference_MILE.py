import argparse
import glob
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as pth_transforms
import faiss

from data.datasets import MVDataset, MILESampler
from dino.dino_args import get_dino_args, get_mile_args
from dino.utils import fix_random_seeds
from model.inference_model import init_model, process_embeddings
from sagemaker.sagemaker_args import get_sagemaker_args
from logger_config import logger

SEED = 1
random.seed(SEED)
fix_random_seeds(SEED)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("DINO_SGM", parents=[get_dino_args(), get_mile_args(), get_sagemaker_args()])
    return parser.parse_args()

def get_checkpoint_path(args: argparse.Namespace) -> Optional[str]:
    if args.ckpt_name is None:
        return None
    
    ckpt_path = os.path.join(args.ckpt_root, args.model_name, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.ckpt_root, args.model_name, "checkpoints", args.ckpt_name)
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    logger.info(f"Using checkpoint: {ckpt_path}")
    return ckpt_path

def get_transforms(args: argparse.Namespace) -> Dict[str, pth_transforms.Compose]:
    inference_transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    empty_transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),
        pth_transforms.ToTensor(),
    ])

    return {"inference": inference_transform, "empty": empty_transform}

def load_datasets(args: argparse.Namespace, transform: pth_transforms.Compose) -> Dict[str, MVDataset]:
    datasets_map = {}
    for mode in glob.glob(os.path.join(args.data_path, "*")):
        key = os.path.basename(mode)
        random_k_samples = MILESampler(
            args.samples_per_class,
            transform,
            no_pad=args.no_pad,
            duplicate_samples=args.duplicate_samples,
            reverse_duplicate_samples=args.reverse_duplicate_samples,
            deterministic=args.deterministic,
            input_order=args.input_order,
            stitching=args.stitching
        )
        dataset = MVDataset(mode, random_k_samples, args.subset_classes)
        datasets_map[key] = dataset
    return datasets_map

def process_embeddings_for_modes(args: argparse.Namespace, model: nn.Module, datasets_map: Dict[str, MVDataset], device: str) -> Dict[str, Dict[str, np.ndarray]]:
    results = {}
    for mode, dataset in datasets_map.items():
        if mode not in [args.target_source, args.test_source]:
            continue
        logger.info(f"Processing embeddings for {mode}")
        results[mode] = process_embeddings(args, model, dataset, args.output_type, device=device)
    return results

def compute_similarity_scores(one: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    distances = []
    for i in range(one.shape[0]):
        sample = one[i, :].unsqueeze(0) if len(one[i, :].shape) == 1 else one[i, :]
        distances.append(similarity(sample, other).unsqueeze(0))
    return torch.cat(distances, dim=0)

def compute_recalls(args: argparse.Namespace, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, cls_type: Tuple[int, int, str], k_max: int) -> Tuple[np.ndarray, float]:
    cosine_index = faiss.IndexFlatIP(X_train.shape[-1])
    cosine_index.add(X_train.cpu())
    S, I = cosine_index.search(X_test.cpu(), k_max)
    classes_list = sorted(list(set([e.item() for e in Y_test])))

    is_covered = np.zeros((len(classes_list), k_max))
    for idx, c in enumerate(classes_list):
        mask = Y_test == c
        I_ = I[mask]
        S_ = S[mask]
        I_max = S_.argmax(axis=0)
        curr_covered = False
        for k in range(k_max):
            sample_idx = I_[I_max[k]][k]
            pred = Y_train[sample_idx]
            if pred == c:
                curr_covered = True
            is_covered[idx][k] = (1 if curr_covered else 0)
    recalls = is_covered.sum(axis=0) / is_covered.shape[0]
    m_recall = np.mean(recalls)
    
    return recalls, m_recall

def get_cls_types(args: argparse.Namespace) -> List[Tuple[int, int, str]]:
    if args.view == "multi-view":
        return [(None, None, "rank_mv_max")]
    else:
        cls_types = [
            (4, 4, "max"),
            (6, 6, "max"),
            (8, 8, "max"),
            (10, 10, "max"),
            (12, 12, "max"),
            (14, 14, "max"),
            (16, 16, "max"),
        ]
        return [c for c in cls_types if c[0] <= args.samples_per_class]

def get_output_path(args: argparse.Namespace, ckpt_path: Optional[str], cls_type: Tuple[int, int, str]) -> str:
    suffix = f"res={args.inference_res}_" if args.inference_res > 224 else ""

    if ckpt_path:
        base, ckpt_name = os.path.split(ckpt_path)
    else:
        base = os.path.join(args.ckpt_root, args.model_name)
        ckpt_name = "vanilla"

    base = os.path.join(base, "test", args.dataset_name, f"{args.view}")
    if args.sample:
        base = os.path.join(base, "sample")

    if cls_type[2] == "rank_mv_max":
        base_ = os.path.join(base, f"out_{args.output_type}_ty2")
        if args.samples_per_class > 4:
            base_ = f"{base_}_S={args.samples_per_class}"
        if args.duplicate_samples > 1:
            base_ = f"{base_}_x{args.duplicate_samples}_Rev={args.reverse_duplicate_samples}"
        if args.no_pad:
            base_ = f"{base_}_NO_PAD"
    else:
        n, m, average_fun = cls_type
        base_ = os.path.join(base, f"S={n}x{m}_{average_fun}")

    os.makedirs(base_, exist_ok=True)
    return os.path.join(
        base_,
        f"report_{args.test_source}_{args.target_source}_k={args.k_max}_{suffix}{ckpt_name.replace('.pth', '')}.json",
    )

def main():
    args = parse_arguments()
    ckpt_path = get_checkpoint_path(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model(args, ckpt_path, 0, device=device, peft=args.peft)

    transforms = get_transforms(args)
    datasets_map = load_datasets(args, transforms["inference"])
    results = process_embeddings_for_modes(args, model, datasets_map, device.type)

    X_train, Y_train = results[args.target_source]["X"], results[args.target_source]["Y"]
    X_test, Y_test = results[args.test_source]["X"], results[args.test_source]["Y"]

    assert X_train.shape[0] == Y_train.shape[0], f"Train shape mismatch: {X_train.shape} vs {Y_train.shape}"
    assert X_test.shape[0] == Y_test.shape[0], f"Test shape mismatch: {X_test.shape} vs {Y_test.shape}"

    logger.info(f"X size train/test: {X_train.shape}, {X_test.shape}")
    logger.info(f"Y size train/test: {Y_train.shape}, {Y_test.shape}")

    cls_types = get_cls_types(args)

    for cls_type in cls_types:
        outfile = get_output_path(args, ckpt_path, cls_type)
        recalls, m_recall = compute_recalls(args, X_train, Y_train, X_test, Y_test, cls_type, args.k_max)

        logger.info(f"Writing results to: {outfile}")
        logger.info(f"Mean recall: {m_recall}")

        with open(outfile, "w") as f:
            json.dump({
                "recalls": [float(e) for e in recalls.tolist()],
                "mean": float(m_recall),
            }, f, indent=2)
            

if __name__ == "__main__":
    main()