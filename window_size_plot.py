import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data = pd.read_csv("window_size_analysis.csv")

X = data[["size"]]
y = data[["precision"]]
y = y * 100
y1 = data[["recall"]]
y1 = y1 * 100

plt.plot(X, y, color="#ff7f0e", linewidth=2, label="Precision")
plt.plot(X, y1, color="#3ca02c", linewidth=2, label="Recall")
plt.xlabel("Step window size (each step 50ms)", fontsize=14)
plt.ylabel("Percentage", fontsize=14)
plt.legend(loc="upper left")
plt.show()