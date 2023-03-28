"""
There are pre-proccessor depends  on the column type
I need to use number_transformer and catagrical_transformer
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

data = pd.read_csv("train.csv")


y = data.SalePrice
x = data.drop(["SalePrice"], axis=1)


number_columns = [col for col in x.columns if x[col].dtype in ["int64", "float64"]]

catagrical_columns = [col for col in x.columns if x[col].dtype == "object"]


x_full = x[number_columns + catagrical_columns].copy()

x_train, x_valid, y_train, y_valid = train_test_split(x_full, y, train_size=0.8, test_size=0.2, random_state=0)

number_transformer = SimpleImputer(strategy="mean")

catagrical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)


pre_proccosser = ColumnTransformer(
    transformers=[
        ("num", number_transformer, number_columns),
        ("cat", catagrical_transformer, catagrical_columns),
    ]
)

pre_proccosser.fit(x_train)
x_valid_transformed = pre_proccosser.transform(x_valid)

model = XGBRegressor(n_estimators=1000, learning_rate=0.05) 

my_pipeline = Pipeline(steps=[("preprocessor", pre_proccosser), ("model", model)])

my_pipeline.fit(x_train, y_train, model__early_stopping_rounds=5, model__eval_set=[(x_valid_transformed, y_valid)], model__verbose=True)


# get mea 
predictions = my_pipeline.predict(x_valid)
print("MAE:", mean_absolute_error(predictions, y_valid))



test_data = pd.read_csv("test.csv")
x_test = test_data[number_columns + catagrical_columns].copy()

predictions = my_pipeline.predict(x_test)

output = pd.DataFrame({"Id": test_data.index, "SalePrice": predictions})
output = pd.DataFrame({"Id": test_data["Id"], "SalePrice": predictions})

output.to_csv("submission.csv", index=False)

