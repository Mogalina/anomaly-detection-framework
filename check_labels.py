import os, sys
import numpy as np

ROOT = os.path.abspath('.')
sys.path.insert(0, os.path.join(ROOT, 'src'))

from tests.utils.smd_loader import load_smd_dataset
train_data, _, _, test_windows, test_labels = load_smd_dataset(seq_len=50, max_train=2000, max_test=1440)
print("Total anomalies in test_labels:", np.sum(test_labels))
print("Test labels length:", len(test_labels))
