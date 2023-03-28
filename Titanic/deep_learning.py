##
import pandas as pd
data = pd.read_csv('./train.csv')
data.head()
##
type(data.describe())
##
data.describe()
##
data.columns
##
data.isna()
##
import seaborn as sns
sns.histplot(data=data,x='HomePlanet',hue='Transported')
##
test_data = pd.read_csv('./test.csv')
passenger_id = test_data['PassengerId']
test_data.head()
##
def fill_missing_values(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col].fillna("unknown",inplace=True)
        else:
            data[col].fillna(0,inplace=True)
##
fill_missing_values(data)
fill_missing_values(test_data)
print(data.isnull().sum())
print(test_data.isnull().sum().sum())
##
data.head()
##
##
data.drop(['Name','PassengerId'],axis=1,inplace=True)
test_data.drop(['Name','PassengerId'],axis=1,inplace=True)

data.head()
##
test_data.head()
##
data.info()
##
float_64_cols = [col for col in data.columns if data[col].dtype == 'float64']
float_64_cols
##
#rows with value inf
for col in float_64_cols:
    data[col] = data[col].replace(float('inf'),10**8).replace(float('nan'),0).round().astype('int')
    test_data[col] = data[col].replace(float('inf'),10**8).replace(float('nan'),0).round().astype('int')
##
data.dtypes
##
test_data.dtypes
##
data.columns
##
data[["Dec","Cabin_num", "side"]] = data['Cabin'].str.split("/", expand=True)
test_data[["Dec","Cabin_num", "side"]] = test_data['Cabin'].str.split("/", expand=True)
##
data.head()
##
test_data.head()
##
data.drop(['Cabin'],axis=1,inplace=True)
test_data.drop(['Cabin'],axis=1,inplace=True)
## lebel econdoig for object columns
data.head()
##
test_data.head()
##
data['CryoSleep'] = data['CryoSleep'].astype(str)
data['VIP'] = data['VIP'].astype(str)

test_data['CryoSleep'] = test_data['CryoSleep'].astype(str)
test_data['VIP'] = test_data['VIP'].astype(str)

##
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])
        test_data[col] = le.transform(test_data[col])
data.head()
##
test_data.head()
##
mat_cor = data.corr()
sns.heatmap(mat_cor,cmap='coolwarm')
##
X = data.drop(['Transported'],axis=1)
Y = data['Transported']
##
from sklearn.model_selection import train_test_split
xtrain,xvalid,ytrain,yvalid = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)
##
print(Y.shape)
print(Y.sum())
##
print(ytrain.shape)
print(ytrain.sum())
##
xtrain.shape
## make gradient boost classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(max_depth = 5)
gbc.fit(xtrain, ytrain)
##
print(xvalid.shape)
gbc.n_features_in_
##
ypred = gbc.predict(xvalid)
ypred
## change ypred to bool
from sklearn.metrics import accuracy_score
accuracy_score(yvalid,ypred)

##
test_data.head()
##
test_data["Cabin_num"] = test_data["Cabin_num"].replace("unknown", '0').astype('int')
##
test_data.head()
##
test_data.dtypes
##
##
ypred_test = gbc.predict(test_data)
##
ypred_test
##
submission = pd.DataFrame({'PassengerId':passenger_id,'Transported':ypred_test})
submission
##
submission.to_csv('sklearn griediet_bost.csv',index=False)
