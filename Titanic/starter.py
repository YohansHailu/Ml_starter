##
import pandas as pd
data = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
data
##
data.dtypes
##
data['A'] = data['A'].astype('float64')
data.dtypes
##
class my_boj:
    def __init__(self, a):
        self.a = a
    def dboule_it(self):
        self.a = self.a * 2
    def __str__(self):
        return str(self.a)
##
data['A'] = data['A'].astype('int64')
data["A"].dtypes
##
