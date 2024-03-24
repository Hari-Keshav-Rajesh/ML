import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.preprocessing import image_dataset_from_directory
from keras.models import load_model
from keras import layers

# Load the data
train_data = image_dataset_from_directory(
    directory='./train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)

valid_data = image_dataset_from_directory(
    directory='./valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)

# Load the model
pretrained_model = load_model('./../cv-course-models/inceptionv1')

# Define the model
model = keras.Sequential([
    pretrained_model,
    layers.Flatten(),
    layers.Dense(6,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

# Compile the model
optimizer = keras.optimizers.Adam(epsilon=0.01)
model.compile(
    optimizer=optimizer,
    loss = 'binary_crossentropy',
    metrics=['binary_accuracy'],
)

# Train the model
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=5,
)

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['binary_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

