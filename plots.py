import numpy as np
import matplotlib.pyplot as plt


preamble = 'c:/users/drago/desktop/research-project/final_runs/'

valid_metric_files = [
    'training_v2_16-06-2024_20-30_0.001_all_50_20/validation_metric_per_epoch.npy',
    'training_v2_16-06-2024_17-00_0.001_2000_50_20/validation_metric_per_epoch.npy',
    'training_v2_19-06-2024_18-00_0.001_200_50_20/validation_metric_per_epoch.npy',
]

valid_metric_files = [preamble + f for f in valid_metric_files]


for file in valid_metric_files:
    valid_scores = np.load(file)
    plt.plot(np.arange(len(valid_scores)), valid_scores)

    plt.show()
