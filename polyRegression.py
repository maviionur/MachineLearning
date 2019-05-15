import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

dataframe = pd.read_csv("./BesiktasGol.csv")

x = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, 3].values
topGol = dataframe.iloc[:, 3].values
sezon = dataframe.iloc[:, 0].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.44, random_state = 0)

pol_reg = PolynomialFeatures(degree=2)  # bu bizim ön işlememiz biz herhangi bir sayıyı bu sayede polinomal bir şekilde
                                        # ifade edebiliriyoruz. Kütüphaneye dikkat edersen sklearn.propossing den import edilior yani ön işleme
x_pol_reg = pol_reg.fit_transform(x)

print(x_pol_reg)
print(y)

# buradaki amacımız öğrenme verilerimizi polinomal veriye çevirip sonra linner modele vermek bu sayede daha başarılı sonuç elde edilmektedir.
# y = b0 + b1*x + b2*x^2 + b3*x^3 .... denklemindeki b katsayılarını y verilerine göre öğrenmek sonrasında tahminde bulunabilmektir.
lin_reg = LinearRegression()
lin_reg.fit(x_pol_reg,y)


a = x_test.tolist()
df2 = pd.DataFrame({'sezon':x_test[:,0]})


sezon = df2.iloc[:, 0:1].values

plt.scatter(sezon,y_test,edgecolors="red")
plt.plot(sezon,lin_reg.predict(pol_reg.fit_transform(x_test)),color="blue")
plt.show()

pol_reg2 = PolynomialFeatures(degree=4) # derece arttırldığında tahmine daha çok yaklaşılmıştır.
x_pol_reg2 = pol_reg2.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_pol_reg2,y)

plt.scatter(sezon,y_test,edgecolors="green")
plt.plot(sezon,lin_reg2.predict(pol_reg2.fit_transform(x_test)),color="blue")
plt.show()

print(lin_reg2.predict(pol_reg2.fit_transform(x_test)))