from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import List

@dataclass
class DataIngestionArtifact:
    train_paths:List[str]
    train_labels:List[int]
    val_paths:List[str]
    val_labels:List[int]

@dataclass
class DataTransformationArtifact:
    train_dataloader:DataLoader
    val_dataloader:DataLoader

