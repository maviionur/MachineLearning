import pandas as pd

dataframe = pd.read_csv("./BesiktasGol.csv")

#dataframe.sort_index(axis=1 ,ascending=True)

data_frame = dataframe.iloc[::-1]
data_frame = data_frame.sort_index(ascending=True, axis=0)
data_frame = data_frame.reindex(index=data_frame.index[::-1])
data_frame.head()

print(dataframe)
#columns = dataframe.columns.tolist()
#columns = columns[-1:] + columns[:-1]

#file = open("./yeni.csv", "w")
#file.write(str(columns))
#file.close()


#print(columns)
