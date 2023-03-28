import pandas as pd

data = pd.DataFrame({'a': [1, 2, 3], 'b': [4,"",""]}, index=['x', 'x', 'x'])

print(data)
