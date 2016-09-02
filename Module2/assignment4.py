import pandas as pd


# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#
# .. your code here ..

df = pd.read_html('http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2')[0]
#print(df)
df.columns = ['RK', 'PLAYER', 'TEAM', 'GP', 'G', 'A', 'PTS', '+/-', 'PIM', 'PTS/G', 'SOG', 'PCT', 'GWG', 'PP_G', 'PP_A', 'SH_G', 'SH_A']
#print(df)
# TODO: Rename the columns so that they match the
# column definitions provided to you on the website
#
# .. your code here ..


# TODO: Get rid of any row that has at least 4 NANs in it
#
# .. your code here ..
df = df.dropna(axis=0, thresh=4)
#print(df)


# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
#
# .. your code here ..
df = df[df['PLAYER']!="PLAYER"]

# TODO: Get rid of the 'RK' column
#
# .. your code here ..
df = df.drop(labels=['RK'], axis=1).reset_index(drop=True)
#print(df)

# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
#
# .. your code here ..



# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric


df.GP = pd.to_numeric(df.GP, errors='coerce')
df.PCT = pd.to_numeric(df.PCT, errors='coerce')
print(df.dtypes)
print(df.loc[15, 'GP']+df.loc[16, 'GP'])
# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.

print(len(df))
print(len(df.PCT.unique()))
