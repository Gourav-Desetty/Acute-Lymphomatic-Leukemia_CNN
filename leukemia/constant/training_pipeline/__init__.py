import os
import sys
import pandas as pd
import numpy as np
import torch, random
from pathlib import Path

TRAINING_DATA_DIR  = "NMC_training_data"
TEST_DATA_PRELIM_DIR  = "test_prelim"
TEST_DATA_FINAL_DIR  = "test_final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_PATH =  random.choice(list(Path("PKG_C_NMC/test_final").glob("*/*.bmp")))
MODEL_PATH = "../Models/leukemia_model_efficientnet_b0_01.pth"

"""
DATA INGESTION REALTED CONSTANTS
"""

DATA_INGESTION_DATASET_NAME = "PKG_C_NMC"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float = 0.2