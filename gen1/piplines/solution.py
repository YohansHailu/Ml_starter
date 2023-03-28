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

# use cross validation


def score_model(model, x, y):
    my_pipeline = Pipeline(steps=[("preprocessor", pre_proccosser), ("model", model)])
    my_pipeline.fit(x, y)
    cv_scores = -1 * cross_val_score(
        my_pipeline, x, y, cv=5, scoring="neg_mean_absolute_error"
    )

    return cv_scores.mean()


data = pd.read_csv("train.csv")


y = data.SalePrice
x = data.drop(["SalePrice"], axis=1)


number_columns = [col for col in x.columns if x[col].dtype in ["int64", "float64"]]

catagrical_columns = [col for col in x.columns if x[col].dtype == "object"]


x_train = x[number_columns + catagrical_columns].copy()


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


model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(
    n_estimators=100, criterion="absolute_error", random_state=0
)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)


best_model = min(
    [model_1, model_2, model_3, model_4, model_5],
    key=lambda m: score_model(m, x_train, y),
)

my_pipeline = Pipeline(steps=[("preprocessor", pre_proccosser), ("model", best_model)])
my_pipeline.fit(x_train, y)
cv_scores = -1 * cross_val_score(
    my_pipeline, x, y, cv=5, scoring="neg_mean_absolute_error"
)

test_data = pd.read_csv("test.csv")
x_test = test_data[number_columns + catagrical_columns].copy()

predictions = my_pipeline.predict(x_test)

output = pd.DataFrame({"Id": test_data.index, "SalePrice": predictions})
output = pd.DataFrame({"Id": test_data["Id"], "SalePrice": predictions})

output.to_csv("submission.csv", index=False)
