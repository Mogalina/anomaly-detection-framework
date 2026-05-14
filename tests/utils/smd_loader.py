import os
import numpy as np

def load_smd_dataset(seq_len=50, max_train=800, max_test=200):
    base_dir = os.path.join(os.path.dirname(__file__), '../data/smd')
    train_data = np.loadtxt(os.path.join(base_dir, 'train.txt'), delimiter=',')
    test_data = np.loadtxt(os.path.join(base_dir, 'test.txt'), delimiter=',')
    test_labels = np.loadtxt(os.path.join(base_dir, 'test_label.txt'))

    def window_data(data, labels=None):
        windows = []
        win_labels = []
        for i in range(len(data) - seq_len):
            windows.append(data[i:i+seq_len])
            if labels is not None:
                win_labels.append(1 if np.sum(labels[i:i+seq_len]) > 0 else 0)
        if labels is not None:
            return np.array(windows, dtype=np.float32), np.array(win_labels, dtype=int)
        return np.array(windows, dtype=np.float32)

    train_windows = window_data(train_data)
    test_windows, test_win_labels = window_data(test_data, test_labels)

    # limit size for benchmarking if requested
    if max_train and len(train_windows) > max_train:
        train_windows = train_windows[:max_train]
    
    # separate test normal and anomalous
    normal_idx = np.where(test_win_labels == 0)[0]
    anom_idx = np.where(test_win_labels == 1)[0]
    test_normal = test_windows[normal_idx]
    test_anomalous = test_windows[anom_idx]
    
    if max_test:
        test_normal = test_normal[:max_test]
        test_anomalous = test_anomalous[:max_test // 4]
        
        # Ensure we have anomalies in the ordered test set
        n_anom = min(len(test_anomalous), max_test // 10) # 10% anomalies
        n_normal = max_test - n_anom
        
        ordered_test_windows = np.concatenate([test_normal[:n_normal], test_anomalous[:n_anom]])
        ordered_test_labels = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_anom, dtype=int)])
        
        # Shuffle them together to simulate realistic streaming
        np.random.seed(42)
        shuffle_idx = np.random.permutation(max_test)
        ordered_test_windows = ordered_test_windows[shuffle_idx]
        ordered_test_labels = ordered_test_labels[shuffle_idx]
    else:
        ordered_test_windows = test_windows
        ordered_test_labels = test_win_labels

    return train_windows, test_normal, test_anomalous, ordered_test_windows, ordered_test_labels
