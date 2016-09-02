import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('ggplot')


df = pd.read_csv('students.data', index_col=0)
"""
#1
print(df.columns)
my_series = df.G3
my_df = df[['G3', 'G2', 'G1']]

my_series.plot.hist(alpha=0.5)
my_df.plot.hist(alpha=0.5)
#2
df.plot.scatter(x='G1', y='G3')
plt.suptitle('Title')
plt.xlabel('xlabel')
plt.ylabel('ylabel')

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Final Grade')
ax.set_ylabel('First Grade')
ax.set_zlabel('Daily Alcohol')

ax.scatter(df.G1, df.G3, df.Dalc, c='r', marker="o")
plt.show()
"""

df.plot.scatter(x='G1', y='Dalc')
plt.suptitle('Title')
plt.xlabel('Final Grade')
plt.ylabel('Daily Alcohol')

plt.show()