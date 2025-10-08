import os
import sys
import pandas as pd
import numpy as np
import torch

TRAINING_DATA_DIR  = "NMC_training_data"
TEST_DATA_PRELIM_DIR  = "test_prelim"
TEST_DATA_FINAL_DIR  = "test_final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
DATA INGESTION REALTED CONSTANTS
"""

DATA_INGESTION_DATASET_NAME = "PKG_C_NMC"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float = 0.2