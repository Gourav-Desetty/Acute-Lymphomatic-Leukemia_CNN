import os, sys
import numpy as np
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from tqdm import tqdm
from constant.training_pipeline import DEVICE
from data.validation import DataValidation
from leukemia.logging.logger import logging
from leukemia.exception.exception import CustomException

class EarlyStopping:
    def __init__(self, patience = 5, min_delta=0.0, restore_best_weights = True):
        try:
            self.patience = patience
            self.min_delta = min_delta  
            self.restore_best_weights = restore_best_weights
            self.best_model = None
            self.counter = 0
            self.best_loss = np.inf
            self.early_stop = False
        except Exception as e:
            raise CustomException(e, sys)

    def __call__(self, val_loss, model):
        try:
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                if self.restore_best_weights:
                    self.best_model = model.state_dict()
            else:
                self.counter +=1
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.restore_best_weights and self.best_model is not None:
                        model.load_state_dict(self.best_model)
        except Exception as e:
            raise CustomException(e, sys)



class Train:
    def __init__(self, data_validation:DataValidation) -> None:
        try:
            self.data_validation = data_validation
        except Exception as e:
            raise CustomException(e, sys)

    def train_step(self, model:nn.Module, 
                    loss_fn:nn.Module,
                    optimizer:torch.optim.Optimizer, 
                    dataloader:torch.utils.data.DataLoader, 
                    device=DEVICE) -> Tuple[float, float, float, float, float]:
        try:
            model.train()
            train_loss = 0.0
            train_acc = 0
            train_recall = 0
            all_preds, all_labels = [], []
            all_probs = []

            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                y_logits = model(X)
                
                loss = loss_fn(y_logits, y)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                y_probs = F.softmax(y_logits, dim=1)
                y_preds = torch.argmax(y_logits, dim=1)
                all_probs.extend(y_probs.detach().cpu().numpy())
                all_preds.extend(y_preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())
                train_acc += self.data_validation.accuracy_fn(y_true=y, y_pred=y_preds)
                train_recall += recall_score(y_true=y, y_pred=y_preds, average='weighted', zero_division=0)


            train_loss /= len(dataloader)
            train_acc /= len(dataloader)
            train_recall /= len(dataloader)
            train_f1 = f1_score(all_labels, all_preds, average='weighted')
            all_probs_positive = [prob[1] for prob in all_probs]
            train_roc_curve = roc_auc_score(all_labels, all_probs_positive)

            return train_loss, train_f1, train_roc_curve, train_acc, train_recall
        except Exception as e:
            raise CustomException(e, sys)


    def train_model(self, model:nn.Module, 
                train_dataloader:torch.utils.data.DataLoader,
                val_dataloader:torch.utils.data.DataLoader,
                epochs=5) -> Dict[str, list]:

        try:
            results = {
                "train_loss": [],
                "train_f1_score": [],
                "val_loss": [],
                "val_f1_score": [],
                "train_roc_auc": [],
                "val_roc_auc": [],
                "train_acc": [],
                "val_acc": [],
                "train_recall": [],
                "val_recall": []
            }

            early_stopping = EarlyStopping(patience=5, min_delta=0.001)

            counts = torch.tensor([3389, 7272], dtype=torch.float32)
            total = counts.sum()
            weights = total / (len(counts) * counts)
            loss_fn = nn.CrossEntropyLoss(weight=weights.to(device=DEVICE))

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            

            for epoch in tqdm(range(epochs)):

                train_loss, train_f1, train_roc_auc, train_acc, train_recall = Train.train_step(self, model=model, 
                                                    loss_fn=loss_fn,
                                                    optimizer=optimizer,
                                                    dataloader=train_dataloader)

                val_loss, val_f1, val_roc_auc, val_acc, val_recall = self.data_validation.val_step(model=model,
                                            loss_fn=loss_fn,
                                            dataloader=val_dataloader)

                scheduler.step()

                print(f"\nEpoch {epoch+1:03d}")
                print(f"{'-'*70}")
                print(f" Train | Loss: {train_loss:.5f} | Acc: {train_acc:.2f}% | "
                    f"F1: {train_f1*100:6.2f}% | Recall: {train_recall*100:6.2f}% | "
                    f"ROC AUC: {train_roc_auc:6.4f}")
                print(f" Valid | Loss: {val_loss:.5f}   | Acc: {val_acc:.2f}% | "
                    f"F1: {val_f1*100:6.2f}% | Recall: {val_recall*100:6.2f}% | "
                    f"ROC AUC: {val_roc_auc:6.4f}")
                print(f"{'-'*70}")



                results["train_loss"].append(train_loss)
                results["train_f1_score"].append(train_f1)
                results["train_roc_auc"].append(train_roc_auc)
                results["train_acc"].append(train_acc)
                results["train_recall"].append(train_recall)
                results["val_loss"].append(val_loss)
                results["val_f1_score"].append(val_f1)
                results["val_roc_auc"].append(val_roc_auc)
                results["val_acc"].append(val_acc)
                results["val_recall"].append(val_recall)

                early_stopping(val_loss, model=model)

                if early_stopping.early_stop:
                    print("Early Stopping triggered")
                    break

            return results
        except Exception as e:
            raise CustomException(e, sys)