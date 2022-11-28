from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_features=4, n_informative=2,random_state=0, shuffle=False)
regr = RandomForestRegressor(random_state=0)
regr.fit(X, y)
print(regr.predict([[1.8, 0.4, 1.0, 2.3]]))
