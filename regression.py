import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

dataframe = pd.read_csv("./BesiktasGol.csv")

x = dataframe.iloc[:, :-1].values

y = dataframe.iloc[:, 3].values
topGol = dataframe.iloc[:, 3].values
sezon = dataframe.iloc[:, 0].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.44, random_state = 0)
print(x_test)
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)

a = x_test.tolist()
df2 = pd.DataFrame({'sezon':x_test[:,0]})


sezon = df2.iloc[:, 0:1].values

#print(sezon)

y_predictions = linear_regressor.predict(x_test)
print(y)
print(y_predictions)

#cm = confusion_matrix(y_test, y_predictions)
plt.scatter(sezon,y_test,color="green")
plt.plot(sezon,linear_regressor.predict(x_test),color="blue")
plt.show()