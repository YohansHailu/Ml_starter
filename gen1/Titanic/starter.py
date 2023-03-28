##
import pandas as pd
import numpy as np
data = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

##
data["rich"] = (np.random.rand(len(data)) > 0.5).astype("bool")
data
##
data["train_test"] = 0
data
##
data = data.replace({False:0, True:1})
data

##
data["useless"] = np.NAN
print(data["useless"].dtypes)
data

##
full = pd.concat([data, data["useless"]], axis=1)
full

##
full.drop(1, axis=0)
