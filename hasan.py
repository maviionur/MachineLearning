import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt

df = pd.read_csv("./ab.csv");
#dataAsya = pd.read_csv("asya.csv");
#dataAfrika = pd.read_csv("afrika.csv");

x = df.iloc[:, 0:-1].values
y = df.iloc[:, 12:13].values

print(df)
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


y_predictions = regressor.predict(x_test)

print(y_predictions)
