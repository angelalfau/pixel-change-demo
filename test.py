# 1. Import Modules
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
plt.style.use("seaborn")

# 2. Generate a 10x10 random integer matrix
data = np.random.rand(10,15)
print("Our dataset is : ",data)

# 3. Plot the heatmap
plt.figure(figsize=(10,15))
heat_map = sns.heatmap( data, linewidth = 1 , annot = True)
plt.title( "HeatMap using Seaborn Method" )
plt.show()