from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import os
import pandas as pd
from General_functions import ML_database, make_training_database



os.chdir("..")
data = data_gathering("IAST-segregated/output")

X, y = make_regression(n_targets = 4, n_features=2, n_informative=2,random_state=0, shuffle=False)
regr = RandomForestRegressor(random_state=0)
regr.fit(X, y)

