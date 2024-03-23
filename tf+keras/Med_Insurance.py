import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from keras import layers, callbacks

from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
insurance_df = pd.read_csv("C:/Users/Hari Keshav Rajesh/Desktop/Computer Projects and resources/Datasets/med_insurance/medical_insurance.csv")

# Split the data into X and Y
X = insurance_df.drop('charges', axis=1)
Y = insurance_df.charges

# Preprocessing
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object' and X[cname].nunique() < 10]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Load data back into X
X = preprocessor.fit_transform(X)

# Imputer for Y
imputer = SimpleImputer(strategy='mean')
Y_imputed = imputer.fit_transform(Y.values.reshape(-1, 1)).ravel()

# Define the Model
input_shape = [X.shape[1]]

def create_model():
    model = keras.Sequential(
    [
        layers.Dense(512, activation='relu', input_shape=input_shape),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(1)
    ]
)

    model.compile(
        optimizer='adam',
        loss='mean_absolute_error',
        metrics=['mean_absolute_error']
    )

    return model

# Define early stopping
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True
)

# Cross Validation
kf = KFold(n_splits=5)

results = []

for train_index, test_index in kf.split(X): #kf.split(X) returns the indices of the train and test sets
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y_imputed[train_index], Y_imputed[test_index]

    model = create_model()
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        batch_size=256,
        epochs=1000,
        callbacks=[early_stopping],
        verbose=0
    )

    mae = model.evaluate(X_test, Y_test)
    results.append(mae)

print("Mean Absolute Error: ", np.mean(results))