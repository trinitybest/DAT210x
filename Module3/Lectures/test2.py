from sklearn.datasets import load_iris
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import andrews_curves

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

data = load_iris()
#print(data)
df = pd.DataFrame(data.data, columns=data.feature_names)

df['target_names'] = [data.target_names[i] for i in data.target]
#print(df)

plt.figure()
parallel_coordinates(df, 'target_names')
plt.show()
#print(df.columns)
#print(df[df['target_names']=='setosa'])
#df[df['target_names']=='setosa'].plot.scatter(x='sepal length (cm)', y='sepal width (cm)')

plt.figure()
andrews_curves(df, 'target_names')
plt.show()