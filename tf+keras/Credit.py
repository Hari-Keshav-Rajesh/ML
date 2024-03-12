import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers, callbacks

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Load the data
credit_df = pd.read_csv("C:/Users/Hari Keshav Rajesh/Desktop/Computer Projects and resources/Datasets/loan_data.csv")

# Split the data into X and Y
X = credit_df.drop('Loan_Status', axis=1)
Y = credit_df.Loan_Status.map(lambda x: 1 if x == 'Y' else 0)

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
    layers.BatchNormalization(),
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
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1,activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

# Define early stopping
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    train_X, train_Y,
    validation_data=(val_X, val_Y),
    batch_size=128,
    epochs=500,
    callbacks=[early_stopping],
    verbose=1
)

# Plot the loss and val_loss
history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['binary_accuracy', 'val_binary_accuracy']].plot()
plt.xlabel('Epochs')
plt.ylabel('Binary Accuracy')
plt.title('Accuracy vs Epochs')
plt.show()

#Evaluate the model
score = model.evaluate(val_X, val_Y, verbose=0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

# Predictions
pred = model.predict(val_X)
print(pred.mean())