import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import os
import math

# Constants
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32

# Data Generators
train_datagen = image.ImageDataGenerator(rescale=1.0 / 255)
validation_datagen = image.ImageDataGenerator(rescale=1.0 / 255)

# Train Data Generator
train_dir = 'C:\\Users\\Lavanya\\OneDrive\\Desktop\\Angular Quiz\\image recognition\\datasets\\train'
val_dir = 'C:\\Users\\Lavanya\\OneDrive\\Desktop\\Angular Quiz\\image recognition\\datasets\\val'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Debug validation data
print("Classes found in training data:", train_generator.class_indices)
print("Classes found in validation data:", validation_generator.class_indices)
print("Number of samples in training data:", train_generator.samples)
print("Number of samples in validation data:", validation_generator.samples)

# Check if validation data is empty
if validation_generator.samples == 0:
    raise ValueError(f"No images found in validation directory: {val_dir}")

# Determine number of classes
num_classes = len(train_generator.class_indices)

# Model Definition
model = keras.models.Sequential([
    keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Steps per Epoch
steps_per_epoch = math.ceil(train_generator.samples / BATCH_SIZE)
validation_steps = math.ceil(validation_generator.samples / BATCH_SIZE)

# Train Model
model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

# Save Model
model.save("my_model.h5")
print("Model saved successfully as my_model.h5")
