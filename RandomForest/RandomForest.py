from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import os
import pandas as pd

def data_gathering(path_to_output):
    data = {}
    outputmaps = os.listdir(path_to_output)
    for outputmap in outputmaps:
        mappath = path_to_output + "/" + str(outputmap)
        
        if os.path.isdir(mappath):
            files = os.listdir(mappath)
            for file in files:
                try:
                    paths =  mappath + "/" + str(file)
                    # print(paths)
                    label = file.split("out")[0]
                    #print(label)
                    # break
                    df = pd.read_table(paths, delimiter = ",")
                    # print(df)
                    #df = df.set_index("pressure")
                    data[label] = df
                    #print(data)
                except:
                    print("ERROR !!!, please check " + file + " \n")

    return data

class MachineLearningInput(): 
    """INPUT: keep this order: name molecule, fraction, name molecule, fraction, ..."""
    def __init__(self, *args):
        for idx, item in enumerate(args):
            setattr(self, "attr{}".format(idx), item)
    
    


a = MachineLearningInput( )
# print(a.attr0)


os.chdir("..")
data = data_gathering("IAST-segregated/output")

X, y = make_regression(n_targets = 4, n_features=2, n_informative=2,random_state=0, shuffle=False)
regr = RandomForestRegressor(random_state=0)
regr.fit(X, y)

