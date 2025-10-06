import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from constant.training_pipeline import DEVICE

class DataValidation:
    @staticmethod
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc

    @staticmethod
    def val_step(model:nn.Module,
                loss_fn:nn.Module,
                dataloader:torch.utils.data.DataLoader,
                device=DEVICE):

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
            all_probs_positive = [prob[1] for prob in all_probs]
            val_roc_curve = roc_auc_score(all_labels, all_probs_positive)

        return val_loss, val_f1, val_roc_curve, val_acc, val_recall
