import os, sys
from pathlib import Path
from typing import Tuple, Dict, List
from leukemia.logging.logger import logging
from leukemia.exception.exception import CustomException 
from torchvision import datasets, transforms
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from leukemia.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        try:
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
        except Exception as e:
            raise CustomException(e, sys)

    def __len__(self):
        try:
            return len(self.image_paths)
        except Exception as e:
            raise CustomException(e, sys)

    def __getitem__(self, index):
        try:
            img_path = self.image_paths[index]
            label = self.labels[index]

            img = Image.open(img_path).convert("RGB")

            if self.transform:
                img = self.transform(img)

            return img, label
        except Exception as e:
            raise CustomException(e, sys)


class DataTransformation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        try:
            logging.info("Initialising train transformation")
            train_transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            logging.info("Initialising val transformation")
            val_transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            return train_transform, val_transform
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_transform, val_transform = self.get_transforms()

            train_dataset = CustomDataset(image_paths=self.data_ingestion_artifact.train_paths,
                                            labels=self.data_ingestion_artifact.train_labels,
                                            transform=train_transform)

            val_dataset = CustomDataset(image_paths=self.data_ingestion_artifact.val_paths,
                                            labels=self.data_ingestion_artifact.val_labels,
                                            transform=val_transform)

            batch_size = 32

            train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            data_transformation_artifact = DataTransformationArtifact(train_dataloader=train_dataloader,
                                                                        val_dataloader=val_dataloader)

            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)