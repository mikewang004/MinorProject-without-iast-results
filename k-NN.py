import numpy as np
from matplotlib import pyplot as plt
from General_functions import make_RASPA_database

data_set_raspa= make_RASPA_database()
 
x_vals=data_set_raspa[:,:-1]
y_vals=data_set_raspa[:,-1]

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x_vals, y_vals ,test_size= 0.1, random_state=0)  

from sklearn.neighbors import KNeighborsRegressor
k=np.arange(1,10)
ps=np.arange(1,6)

for p in ps:
    correct=[]
    for n in k:
        # knn=KNeighborsRegressor(n_neighbors=n,metric='cosine',algorithm='brute',n_jobs=-1) 
        knn=KNeighborsRegressor(n_neighbors=n,metric='minkowski',p=p)
        knn.fit(x_train, y_train)
        y_pred=knn.predict(x_test)
        rel_err=np.abs(y_pred-y_test)/y_test
        # print(rel_err)
        mask =rel_err<0.25
        correct.append(len(y_test[mask]))
    
    plt.figure()
    plt.title(f'K nearest meighbors performance for Minkowski metric,p={p}')
    plt.xlabel('k neighbors')
    plt.ylabel('Correct predictions [%]')
    # plt.hlines(len(y_test),xmin=1,xmax=10,color='r')
    plt.plot(k,np.array(correct)/len(y_test)*100,'ko')
    plt.show()