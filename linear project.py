

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

a = pd.read_csv("kn.csv")
print(a)

print(a.shape)
print(a['bedroom'].describe())


c = a.corr()
print(c)

print(a.head())

x = a['bedroom']
y = a['price']
print(x)
print(y)

x = np.array(x).reshape(-1,1)
print(x)

o = LinearRegression()
lm = o.fit(x, y)

print(lm)
print(lm.intercept_)
print(lm.coef_)

b = lm.predict(x)
print(b)

bedroom = []
for x in range(3):
    a = int(input('enter bedroom :'))
    bedroom.append(a)

bedroom = np.array(bedroom).reshape(-1,1)

print(lm.predict(bedroom))
