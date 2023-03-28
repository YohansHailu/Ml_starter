import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

def score_model(model, X_t, X_v, y_t, y_v):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

def choose_the_best_model(x_train, x_valid, y_train, y_valid):

    model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
    model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
    model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
    model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
    model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

    models = [model_1, model_2, model_3, model_4, model_5]




    for model in models:
        mae = score_model(model, x_train, x_valid, y_train, y_valid)
        print("Model MAE: {}".format(mae))

    best_model = min(models, key=lambda model: score_model(model, x_train, x_valid, y_train, y_valid))

    return best_model


# Read the data
data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

x = data.select_dtypes(exclude=['object'])
x_test = test_data.select_dtypes(exclude=['object'])


y = x.SalePrice
x = x.drop(['SalePrice'], axis=1)


x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)



imputer = SimpleImputer()
imputed_x_train = pd.DataFrame(imputer.fit_transform(x_train))
imputed_x_valid = pd.DataFrame(imputer.transform(x_valid))
imputed_x_test = pd.DataFrame(imputer.transform(x_test))

print("   after imputing",len([col for col in x_train.columns if x_train[col].isnull().any()]))


my_model = RandomForestRegressor(n_estimators=100, random_state=0)
score = score_model(my_model, imputed_x_train, imputed_x_valid, y_train, y_valid) 

my_model.fit(imputed_x_train, y_train)

predictions = my_model.predict(imputed_x_test)

output = pd.DataFrame({'Id': test_data.index, 'SalePrice': predictions})
output = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions})

output.to_csv('submission.csv', index=False)
