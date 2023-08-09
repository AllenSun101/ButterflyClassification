import tensorflow as tf
import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Process Data

data_folder = "Data/train"
label_file = pd.read_csv("Data/Training_set.csv")

# Prepare data in training, validation, and testing

data = []
labels = []

for file_name in os.listdir(data_folder):
    image_path = os.path.join(data_folder, file_name)
    img = Image.open(image_path)


    # Resize image
    img = img.resize((128, 128))
    
    img_array = np.array(img) / 255.0
    data.append(img_array)

    # Handle labels
    label_row = label_file[label_file['filename'] == file_name]

    labels.append(label_row["label"].iloc[0])

data = np.array(data)
labels = np.array(labels)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


# Split into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.8, random_state=42)


# Set up model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(75, activation=tf.nn.softmax)
])

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train Model with Validation split
model.fit(train_data, train_labels, epochs=25, validation_split=0.2)

# Model Evaluation
test_loss = model.evaluate(test_data, test_labels)
