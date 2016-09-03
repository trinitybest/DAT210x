import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pandas.tools.plotting import parallel_coordinates

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
df = pd.read_csv('Datasets/wheat.data')
df = df.drop(labels=['id', 'area', 'perimeter'], axis = 1)


#
# TODO: Drop the 'id', 'area', and 'perimeter' feature
# 
# .. your code here ..



#
# TODO: Plot a parallel coordinates chart grouped by
# the 'wheat_type' feature. Be sure to set the optional
# display parameter alpha to 0.4
# 
# .. your code here ..
plt.figure()
parallel_coordinates(df, 'wheat_type', alpha=0.4 )
df.plot.scatter(x='width', y='length')

df.plot.scatter(x='width', y='asymmetry')

plt.show()


