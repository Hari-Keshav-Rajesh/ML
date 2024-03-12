import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

soc_df = pd.read_csv("C:/Users/Hari Keshav Rajesh/Desktop/Computer Projects and resources/Datasets/dummy_data.csv")

ord_encoder  = OrdinalEncoder()
imputer = SimpleImputer(strategy='mean')

soc_df["gender_encoded"] = ord_encoder.fit_transform(soc_df[["gender"]])

soc_df["platform_encoded"] = ord_encoder.fit_transform(soc_df[["platform"]])

soc_df["interests_encoded"] = ord_encoder.fit_transform(soc_df[["interests"]])

soc_df["location_encoded"] = ord_encoder.fit_transform(soc_df[["location"]])

soc_df["demographics_encoded"] = ord_encoder.fit_transform(soc_df[["demographics"]])

soc_df["profession_encoded"] = ord_encoder.fit_transform(soc_df[["profession"]])

soc_df["indebt_encoded"] = ord_encoder.fit_transform(soc_df[["indebt"]])

soc_df["isHomeOwner_encoded"] = ord_encoder.fit_transform(soc_df[["isHomeOwner"]])

soc_df["OwnsCar_encoded"] = ord_encoder.fit_transform(soc_df[["Owns_Car"]])

soc_temp = soc_df[['age','gender_encoded','time_spent','platform_encoded','interests_encoded','location_encoded','demographics_encoded','profession_encoded','income','indebt_encoded','isHomeOwner_encoded','OwnsCar_encoded']]

soc = pd.DataFrame(imputer.fit_transform(soc_temp), columns=soc_temp.columns)

X = soc[soc["time_spent"] >= 3]
X = X.drop("income",axis=1)

Y = soc[soc["time_spent"] < 3]
Y = Y.drop("income",axis=1)

Z1 = soc.loc[soc['time_spent'] >= 3, 'income']
Z2 = soc.loc[soc['time_spent'] < 3, 'income']

train_X,val_X,train_Z1,val_Z1 = train_test_split(X,Z1,random_state=1)

train_Y,val_Y,train_Z2,val_Z2 = train_test_split(Y,Z2,random_state=1)

soc_model1 = RandomForestRegressor(random_state=1)

soc_model2 = RandomForestRegressor(random_state=1)

soc_model1.fit(train_X,train_Z1)
soc_model2.fit(train_Y,train_Z2)

soc_preds1 = soc_model1.predict(val_X)
soc_preds2 = soc_model2.predict(val_Y)

soc_gt3 = soc_preds1.mean()
soc_lt3 = soc_preds2.mean()

print(mean_absolute_error(val_Z1,soc_preds1))
print(mean_absolute_error(val_Z2,soc_preds2))