import os
import glob
import random
import logging
from collections import defaultdict
from typing import List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MVDataset(Dataset):
    def __init__(self, data_root: str, transform: Callable, subset_classes: Optional[int] = None):
        self.transform = transform
        self.data_root = data_root
        self.class_to_samples: List[Tuple[str, List[str]]] = []
        self.class_index: dict = {}

        logging.info(f"Reading data from {self.data_root}")
        classes_list = self._get_filtered_classes(subset_classes)
        logging.info(f"First 10 classes: {', '.join(classes_list[:10])}")

        for class_id in classes_list:
            class_dir = os.path.join(self.data_root, class_id)
            if not os.path.isdir(class_dir):
                continue
            samples = [glob.glob(os.path.join(class_dir, ext)) for ext in ["*.jpg", "*.png"]]
            samples = [e for source in samples for e in source]
            if len(samples) == 0:
                continue
            self.class_to_samples.append((class_id, samples))
            self.class_index[class_id] = len(self.class_index)
        
        logging.info(f"Found {len(self.class_index)} classes")

    def _get_filtered_classes(self, subset_classes: Optional[int]) -> List[str]:
        classes_list = list(os.listdir(self.data_root))
        if subset_classes is None:
            return classes_list

        super_counts = defaultdict(int)
        classes_list = sorted(classes_list)
        filtered_classes_list = []
        for c in classes_list:
            super_class = c.split("_")[1]
            super_counts[super_class] += 1
            if super_counts[super_class] > subset_classes:
                continue
            filtered_classes_list.append(c)
        
        logging.info(f"Classes pool {len(classes_list)} filtered to {len(filtered_classes_list)}")
        return filtered_classes_list

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], int]:
        label, samples = self.class_to_samples[index]
        crops = self.transform(samples)
        label = self.class_index[label]
        return crops, label

    def __len__(self) -> int:
        return len(self.class_to_samples)


class MILESampler:
    def __init__(
        self,
        samples_per_class: int,
        transform: Callable,
        no_pad: bool = False,
        duplicate_samples: int = 1,
        reverse_duplicate_samples: bool = True,
        deterministic: bool = True,
        input_order: int = -1,
        stitching: bool = True
    ):
        self.samples_per_class = samples_per_class
        self.transform = transform
        self.no_pad = no_pad
        self.duplicate_samples = duplicate_samples
        self.reverse_duplicate_samples = reverse_duplicate_samples
        self.deterministic = deterministic
        self.input_order = input_order
        self.stitching = stitching        
        logging.info(f"Reverse duplicate samples: {reverse_duplicate_samples}")

    def __call__(self, samples: List[str]) -> Optional[List[torch.Tensor]]:
        samples = self._select_samples(samples)
        if not samples:
            return None

        samples = self._pad_samples(samples)
        samples = self._reorder_samples(samples)
        samples = self._duplicate_samples(samples)
        samples_crops = [self.transform(sample) for sample in samples]

        if self.stitching:
            return [self._stitch_samples(samples_crops)]
        return samples_crops

    def _select_samples(self, samples: List[str]) -> List[Image.Image]:
        if len(samples) > self.samples_per_class:
            if self.deterministic:
                samples = sorted(samples, key=lambda e: os.path.basename(e))[:self.samples_per_class]
            else:
                samples = random.sample(samples, self.samples_per_class)
        samples = [Image.open(sample) for sample in samples]
        return [s for s in samples if s.mode == "RGB"]

    def _pad_samples(self, samples: List[Image.Image]) -> List[Image.Image]:
        if len(samples) < self.samples_per_class and not self.no_pad:
            assert self.samples_per_class < 100, "Can't feed more than 100 samples"
            copy_samples = [random.choice(samples) for _ in range(self.samples_per_class - len(samples))]
            samples.extend(copy_samples)
        return samples

    def _reorder_samples(self, samples: List[Image.Image]) -> List[Image.Image]:
        if self.input_order == -1:
            return samples[::-1]
        elif self.input_order == 1:
            return samples[0::2] + samples[1::2]
        elif self.input_order == 2:
            return samples[1::2] + samples[0::2]
        elif self.input_order == 3:
            return [samples[1], samples[0], samples[3], samples[2]]
        elif self.input_order == 0:
            return samples
        else:
            raise ValueError("Input order must be 0, 1, 2, 3, or -1")

    def _duplicate_samples(self, samples: List[Image.Image]) -> List[Image.Image]:
        if self.duplicate_samples > 1:
            copy_samples = samples[:]
            for _ in range(self.duplicate_samples - 1):
                if self.reverse_duplicate_samples:
                    copy_samples = copy_samples[::-1]
                samples.extend(copy_samples[1:])
        return samples

    def _stitch_samples(self, samples_crops: List[torch.Tensor]) -> torch.Tensor:
        group_shape = samples_crops[0].shape[1]
        canvas = torch.zeros(3, group_shape * 2, group_shape * 2)
        canvas[:, :group_shape, :group_shape] = samples_crops[0]
        canvas[:, :group_shape, group_shape:] = samples_crops[1]
        canvas[:, group_shape:, group_shape:] = samples_crops[2]
        canvas[:, group_shape:, :group_shape] = samples_crops[3]
        return transforms.Resize((group_shape, group_shape))(canvas)