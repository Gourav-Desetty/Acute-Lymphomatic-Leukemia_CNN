import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from leukemia.constant.training_pipeline import DEVICE
from leukemia.logging.logger import logging
from leukemia.exception.exception import CustomException 
from leukemia.entity.artifact_entity import DataTransformationArtifact
# from leukemia.entity.artifact_entity import DataValidationArtifact

class DataValidation:
    def __init__(self, data_transformation_artifact=DataTransformationArtifact):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.train_dataloader = data_transformation_artifact.train_dataloader
            self.val_dataloader = data_transformation_artifact.val_dataloader
            logging.info("DataValidation initialized")
        except Exception as e:
            raise CustomException(str(e), e)
    @staticmethod
    def accuracy_fn(y_true, y_pred):
        try:
            # logging.info("Calculating accuracy")
            correct = torch.eq(y_true, y_pred).sum().item()
            acc = (correct / len(y_pred)) * 100
            # logging.info(f"Batch accuracy: {acc:.2f}%")
            return acc
        except Exception as e:
            raise CustomException(str(e), e)

    def val_step(self, model:nn.Module,
                loss_fn:nn.Module,
                dataloader:torch.utils.data.DataLoader,
                device=DEVICE):
        try:
            logging.info("Starting validation step")
            model.eval()
            val_loss = 0.0
            val_acc, val_recall = 0, 0
            all_preds, all_labels = [], []
            all_probs = []

            with torch.inference_mode():
                for X, y in dataloader:
                    X, y = X.to(device), y.to(device)

                    val_logits = model(X)
                    loss = loss_fn(val_logits, y)
                    val_loss += loss.item()

                    val_probs = F.softmax(val_logits, dim=1)
                    val_preds = torch.argmax(val_logits, dim=1)
                    all_probs.extend(val_probs.detach().cpu().numpy())
                    all_preds.extend(val_preds.cpu().tolist())
                    all_labels.extend(y.cpu().tolist())
                    val_acc += DataValidation.accuracy_fn(y_true=y, y_pred=val_preds)
                    val_recall += recall_score(y_true=y, y_pred=val_preds, average='weighted', zero_division=0)

                val_loss /= len(dataloader)
                val_acc /= len(dataloader)
                val_recall /= len(dataloader)
                val_f1 = f1_score(all_labels, all_preds, average='weighted')
                all_probs_positive = [float(prob[1]) for prob in all_probs]
                val_roc_curve = float(roc_auc_score(all_labels, all_probs_positive))

            logging.info(f"Validation complete - Loss: {val_loss:.4f} | "
                    f"Acc: {val_acc:.2f}% | F1: {val_f1*100:.2f}% | "
                    f"Recall: {val_recall*100:.2f}% | ROC-AUC: {val_roc_curve:.4f}")

            return float(val_loss), float(val_f1), float(val_roc_curve), float(val_acc), float(val_recall)
        except Exception as e:
            raise CustomException(str(e), e)