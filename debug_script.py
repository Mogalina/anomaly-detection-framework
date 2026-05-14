import numpy as np

threshold = 0.0159
for _ in range(10):
    threshold *= (1 - 0.1)
    print(threshold)
