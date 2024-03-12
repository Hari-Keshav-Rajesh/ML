import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

credit_df = pd.read_csv("C:/Users/Hari Keshav Rajesh/Desktop/Computer Projects and resources/Datasets/loan_data.csv")

Y = credit_df.Loan_Status.map(lambda x: 1 if x == 'Y' else 0)
X_temp = credit_df
X_temp.drop(['Loan_Status'], axis=1, inplace=True)

X_train_full, X_valid_full, Y_train, Y_valid = train_test_split(X_temp, Y, train_size=0.8, test_size=0.2, random_state=0)

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

cateogorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == 'object']

my_cols = numerical_cols + cateogorical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

numerical_transformer = SimpleImputer(strategy='constant')

cateogorical_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numerical_transformer, numerical_cols),
        ('cat', cateogorical_transformer, cateogorical_cols)
    ]
)

model = XGBRegressor(n_estimators=300)

credit = Pipeline(
    steps = [
        ('preprocessor', preprocessor),
        ('model', model)
    ]
)

credit.fit(X_train, Y_train)

preds = credit.predict(X_valid)

MAE = mean_absolute_error(Y_valid, preds)

print("Mean Absolute Error: ", MAE)

plt.figure(dpi=100)

plt.plot(Y_valid, preds,linestyle='-',label='Loan Status')

plt.xlabel('Actual Loan Status')
plt.ylabel('Predicted Loan Status')
plt.show()