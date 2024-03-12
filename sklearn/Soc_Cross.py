import pandas as pd

from sklearn.model_selection import cross_val_score, cross_val_predict
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

soc_df = pd.read_csv("C:/Users/Hari Keshav Rajesh/Desktop/Computer Projects and resources/Datasets/dummy_data.csv")

Y = soc_df['income']
X = soc_df.drop(['income'], axis=1)

numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and X[cname].dtype == "object"]

my_cols = numerical_cols + categorical_cols
X = X[my_cols].copy()

numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

soc = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ]
)

soc.fit(X, Y)

scores = -1 * cross_val_score(soc, X, Y, cv=5, scoring='neg_mean_absolute_error')

predict = cross_val_predict(soc, X, Y, cv=5)

print(f"MAE: {scores.mean()}\nPredictions: {predict.mean()}")