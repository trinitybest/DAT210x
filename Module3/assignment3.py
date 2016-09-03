import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
df = pd.read_csv('Datasets/wheat.data')



fig = plt.figure()
a = fig.add_subplot(111, projection='3d')
a.scatter(df.area, df.perimeter, df.asymmetry, c='red')
#
# TODO: Create a new 3D subplot using fig. Then use the
# subplot to graph a 3D scatter plot using the area,
# perimeter and asymmetry features. Be sure to use the
# optional display parameter c='red', and also label your
# axes
# 
# .. your code here ..


fig = plt.figure()
b = fig.add_subplot(111, projection='3d')
b.set_xlabel('width')
b.set_ylabel('groove')
b.set_zlabel('length')
b.scatter(df.width, df.groove, df.length , c='red')
#
# TODO: Create a new 3D subplot using fig. Then use the
# subplot to graph a 3D scatter plot using the width,
# groove and length features. Be sure to use the
# optional display parameter c='green', and also label your
# axes
# 
# .. your code here ..


plt.show()


