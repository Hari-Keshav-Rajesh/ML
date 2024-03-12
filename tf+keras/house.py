import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers, callbacks

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the data

data = pd.read_csv("C:/Users/Hari Keshav Rajesh/Desktop/Computer Projects and resources/Datasets/NY-House-Dataset.csv")

# Split the data into X and Y
X = data.drop('PRICE',axis=1)
Y = data.PRICE

# Train test split
train_X, val_X, train_Y, val_Y = train_test_split(X, Y, random_state=0)

# Preprocessing
numerical_cols = [cname for cname in train_X.columns if train_X[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in train_X.columns if train_X[cname].dtype == 'object' and train_X[cname].nunique() < 10]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Load data back into X
train_X = preprocessor.fit_transform(train_X)
val_X = preprocessor.transform(val_X)

# Define the model
input_shape = [train_X.shape[1]]
model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='mae'
)

# Define the callbacks
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    train_X, train_Y,
    validation_data=(val_X, val_Y),
    batch_size=64,
    epochs=50,
    callbacks=[early_stopping],
    verbose=1
)

# Plot the learning curves
history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.show()

# Predictions
pred = model.predict(val_X)
# Reverse Batch Normalization
pred = pred * val_Y.std() + val_Y.mean()
print(pred.mean())