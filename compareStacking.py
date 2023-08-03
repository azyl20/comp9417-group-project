import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# the following hardcoded data is obtained from the output of other files
# This file is just for visualisation

labels = ["RF\nHGBC\nKNN", "RF\nHGBC\nLog.", "RF\nHGBC\nSVM",
          "RF\nHGBC\nKNN\nSVM", "RF\nHGBC\nSVM\nLog.", "RF\nHGBC"]

data = [0.9389759036144589, 0.9343373493975906, 0.9278915662650603,
        0.932710843373495, 0.9312048192771089, 0.9265662650602416]

plt.bar(labels, data)
plt.title("Comparison of Stacking Models")
plt.ylabel("test score")
plt.ylim(0.92, 0.95)
plt.savefig("results/CompareStacking.png")

plt.show()
