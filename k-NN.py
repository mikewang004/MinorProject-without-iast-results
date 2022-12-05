import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from General_functions import ML_database

chemstructure=ML_database()
#data = data_gathering("Raspa/outputs")
data_set= pd.read_csv('Raspa/outputs/ML_data_fractions.csv')
#data_set=np.loadtxt('Raspa/outputs/ML_data_fractions.csv',skiprows=1)
# for i,m in enumerate(data_set['molecule']):
#     data_set['molecule'][i]=chemstructure[m]
    
x_vals=data_set.iloc[:,range(53)].values
y_vals=data_set.iloc[:,-1].values

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x_vals, y_vals ,test_size= 0.25, random_state=0)  

# feature Scaling  
# from sklearn.preprocessing import StandardScaler    
# st_x= StandardScaler()    
# x_train= st_x.fit_transform(x_train)    
# x_test= st_x.transform(x_test)

from sklearn.neighbors import KNeighborsRegressor
k=np.arange(1,10)
correct=[]
for n in k:
    knn=KNeighborsRegressor(n_neighbors=n, metric='minkowski', p=2) 
    knn.fit(x_train, y_train)
    y_pred=knn.predict(x_test) 
    rel_err=np.abs(y_pred-y_test)/y_test
    print(rel_err)
    mask =rel_err<0.1
    correct.append(len(y_test[mask]))

plt.xlabel('k neighbors')
plt.ylabel('Correct predictions [%]')
# plt.hlines(len(y_test),xmin=1,xmax=10,color='r')
plt.plot(k,np.array(correct)/len(y_test)*100,'ko')
# k=3 #test performance of different k's 

# #say we have all our previous datapoints in a set X, we compute the k nearest neighbors for a new point as
# new_data = np.array([5,3,4,6,7]) 
# distance= la.norm(X-new_data,axis=1)
# k_nearest = distance.argsort()[:k]

# nearest_data = X[k_nearest].mean() #for a regression model, the mean of the nearest neighbors is taken as output

# #sk.neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=1, metric='minkowski')