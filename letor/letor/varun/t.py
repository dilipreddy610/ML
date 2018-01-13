import seaborn as snp
import matplotlib.pyplot as plt
import numpy as np
from dask.dataframe.core import DataFrame


#dict = np.array([3, 4, 5, 6], [0.879,0.345,0.567,0.456])
dict = {3:0.879, 4:0.345, 5:0.567, 6:0.456}
#plt.plot([3, 4, 5, 6], [0.879,0.345,0.567,0.456])
plt.plot(list(dict.keys()), list(dict.values()))
plt.xlabel('K')
plt.ylabel('Erms on validation data')
plt.show()
