import numpy as np
import pandas as pd


W1 = pd.read_csv("train/W1.csv")
W1 = np.array(W1)
# b1 = pd.read_csv("train/b1.csv")
# b1 = np.array(b1)
# W2 = pd.read_csv("train/W2.csv")
# W2 = np.array(W2)
# b2 = pd.read_csv("train/b2.csv")
# b2 = np.array(b2)

print(W1[:, 1:].shape)
