import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error



house_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

feature_names = ["LotArea",
"YearBuilt",
"1stFlrSF",
"2ndFlrSF",
"FullBath",
"BedroomAbvGr",
"TotRmsAbvGrd",
"OverallQual",
'YrSold',
]

X = house_data[feature_names]
y = house_data.SalePrice


house_model = RandomForestRegressor(random_state=1) 
house_model.fit(X,y)

test_X = test_data[feature_names]

predictions = house_model.predict(test_X)

test_data["SalePrice"] =  predictions 

solution = test_data[["Id","SalePrice"]]

solution.to_csv("solution.csv", index = False)

