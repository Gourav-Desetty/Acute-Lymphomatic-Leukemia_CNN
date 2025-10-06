import os, sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
from leukemia.logging.logger import logging
from leukemia.exception.exception import CustomException 
from leukemia.entity.artifact_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self):
        self.data_path = Path("PKG_C_NMC")
        self.train_dir = self.data_path / "C-NMC_training_data"
        self.test_prelim_dir = self.data_path / "test_prelim"
        self.test_final_dir = self.data_path / "test_final"


    def load_data(self, data_dir: Path) -> Tuple[List[str], List[int], List[str]]:
        try:
            images_paths, labels, subject_ids = [], [], []

            for fold_name in os.listdir(data_dir):
                fold_path = os.path.join(data_dir, fold_name)
                if not os.path.isdir(fold_path):
                    continue


                # Load cancer cells - 'all'
                logging.info("Loading cancer cells - all")
                all_folder = os.path.join(fold_path, "all")
                if os.path.exists(all_folder):
                    for file_name in os.listdir(all_folder):
                        if file_name.endswith(('.bmp')):
                            images_paths.append(os.path.join(all_folder, file_name))
                            labels.append(1)

                            parts = file_name.split('_')
                            subject_id = parts[1] if len(parts) > 1 else "unknown"
                            subject_ids.append(f"cancer_{subject_id}")

                # Load normal cells - 'hem'
                logging.info("Loading normal cells - hem")
                hem_folder = os.path.join(fold_path, "hem")
                if os.path.exists(hem_folder):
                    for file_name in os.listdir(hem_folder):
                        if file_name.endswith(('.bmp')):
                            images_paths.append(os.path.join(hem_folder, file_name))
                            labels.append(0)

                            parts = file_name.split('_')
                            subject_id = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else "unknown"
                            subject_ids.append(f"normal_{subject_id}")

            logging.info("retrieving image paths, labels and subject ids")
            return images_paths, labels, subject_ids

        except Exception as e:
            raise CustomException(e, sys)
    
    def split_data(self, image_paths: List[str], labels: List[int], subject_ids: List[str], test_size=0.2):
        try:
            unique_subjects = list(set(subject_ids))

            logging.info("Starting train test split")
            train_subjects, val_subjects = train_test_split(unique_subjects, 
                                                            test_size=test_size, 
                                                            random_state=42, 
                                                            stratify= [s.split('_')[0] for s in unique_subjects])
            logging.info("performed train test split")
            train_paths, train_labels = [], []
            val_paths, val_labels = [], []

            for path, label, subject in zip(image_paths, labels, subject_ids):
                if subject in train_subjects:
                    train_paths.append(path)
                    train_labels.append(label)
                else:
                    val_paths.append(path)
                    val_labels.append(label)

            return train_paths, train_labels, val_paths, val_labels

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            path = self.train_dir
            images_paths, labels, subject_ids = self.load_data(path)
            train_paths, train_labels, val_paths, val_labels = self.split_data(image_paths=images_paths,
                                                                                labels=labels,
                                                                                subject_ids=subject_ids)
            
            data_ingestion_artifact = DataIngestionArtifact(train_paths=train_paths,
                                                            train_labels=train_labels,
                                                            val_paths=val_paths,
                                                            val_labels=val_labels)

            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)