import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt



filename = 'yeast3\yeast3_label_class.csv_Weighted.csv'

df = pd.read_csv(filename)

g = pd.plotting.scatter_matrix(df)

plt.show()


