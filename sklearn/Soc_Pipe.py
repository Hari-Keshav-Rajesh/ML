import pandas as pd

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

soc_df = pd.read_csv("C:/Users/Hari Keshav Rajesh/Desktop/Computer Projects and resources/Datasets/dummy_data.csv")

X = soc_df
Y = soc_df['income']

trainX, valX, trainY, valY = train_test_split(X, Y, random_state=1)

categorical_cols = [cname for cname in trainX.columns if trainX[cname].nunique() < 10 and trainX[cname].dtype == "object"]

numerical_cols = [cname for cname in trainX.columns if trainX[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
train_X = trainX[my_cols].copy()
val_X = valX[my_cols].copy()

numerical_transofrmer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transofrmer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
#model = RandomForestRegressor(n_estimators=700,max_depth=1,random_state=0)

soc = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

soc.fit(train_X, trainY)

soc_preds = soc.predict(val_X)

MAE = mean_absolute_error(valY, soc_preds)

print(f"MAE: {MAE}")