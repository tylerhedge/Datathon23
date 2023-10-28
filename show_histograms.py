import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('TDHospital/TD_HOSPITAL_TRAIN.csv')
#dataset.dropna(inplace=True)


for i, x in enumerate(dataset.columns):
  data = dataset[x].dropna()
  plt.hist(data, bins='auto', edgecolor='black', alpha=0.7, color='blue')
  plt.xlabel(x)
  plt.ylabel('Frequency')
  plt.title(f"Histogram for {x}")
  plt.show()