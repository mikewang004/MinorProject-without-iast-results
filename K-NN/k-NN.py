import numpy as np
import numpy.linalg as la

k=3 #test performance of different k's 

#say we have all our previous datapoints in a set X, we compute the k nearest neighbors for a new point as
new_data = np.array([5,3,4,6,7]) 
distance= la.norm(X-new_data,axis=1)
k_nearest = distance.argsort()[:k]

nearest_data = X[k_nearest].mean() #for a regression model, the mean of the nearest neighbors is taken as output
