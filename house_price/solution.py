import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

X_full = pd.read_csv('./train.csv', index_col='Id')
X_test_full = pd.read_csv('./test.csv', index_col='Id')

# see if there are missing values in X 

y = X_full.SalePrice
feutures = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']


X = X_full[feutures].copy()
X_test = X_test_full[feutures].copy()

missing_val_count_by_column = (X.isnull().sum().sum())
print(missing_val_count_by_column)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]


def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)


for model in models:
    mae = score_model(model)
    print("Model MAE: {}".format(mae))

best_model = min(models, key=score_model)

my_model = best_model

my_model.fit(X, y)

predictions = my_model.predict(X_test)


output = pd.DataFrame({'Id': X_test.index, 'SalePrice': predictions})
output.to_csv('submission.csv', index=False)
