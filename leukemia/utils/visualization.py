import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from PIL import Image
import torch
from torchvision import transforms

def plot_graph(results: Dict[str, List[float]]):
    train_loss = results["train_loss"]
    val_loss = results["val_loss"]
    train_f1_score = results["train_f1_score"]
    val_f1_score = results["val_f1_score"]
    train_roc_auc = results["train_roc_auc"]
    val_roc_auc = results["val_roc_auc"]
    train_acc = results["train_acc"]
    val_acc = results["val_acc"]

    epochs = list(range(len(results["train_loss"])))

    fig, axs = plt.subplots(1, 4, figsize=(22, 5))

    # Loss 
    axs[0].plot(epochs, train_loss, label="Train Loss")
    axs[0].plot(epochs, val_loss, label="Val Loss")
    axs[0].set_title("Loss Over Epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    
    # Accuracy 
    axs[1].plot(epochs, train_acc, label="Train Acc")
    axs[1].plot(epochs, val_acc, label="Val Acc")
    axs[1].set_title("Accuracy Over Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    # F1 Score 
    axs[2].plot(epochs, train_f1_score, label="Train F1")
    axs[2].plot(epochs, val_f1_score, label="Val F1")
    axs[2].set_title("F1 Score Over Epochs")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("F1 Score")
    axs[2].legend()

    # ROC-AUC 
    axs[3].plot(epochs, train_roc_auc, label="Train ROC-AUC")
    axs[3].plot(epochs, val_roc_auc, label="Val ROC-AUC")
    axs[3].set_title("ROC-AUC Over Epochs")
    axs[3].set_xlabel("Epochs")
    axs[3].set_ylabel("ROC-AUC Score")
    axs[3].legend()

    plt.tight_layout()
    plt.show()