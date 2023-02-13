# Step 1: Data Collection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data into a pandas dataframe
data = pd.read_csv("emotion_data.csv")

# Step 2: Pre-processing
from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['image'], data['emotion'], test_size=0.2)

# Step 3: Feature extraction
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Use ImageDataGenerator to pre-process the images
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate the training and test datasets
train_generator = train_datagen.flow_from_directory(X_train, y_train, target_size=(150,150), batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(X_test, y_test, target_size=(150,150), batch_size=32, class_mode='categorical')

# Step 4: Model selection
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Initialize a sequential model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Add a flatten layer
model.add(Flatten())

# Add dense layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 5: Model training
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=50, validation_data=test_generator, validation_steps=50)

# Step 6: Model evaluation
score = model.evaluate_generator(test_generator, steps=50)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Step 7: Hyperparameter tuning
# You can try adjusting the number of layers, layer sizes, activation functions, and other hyperparameters
# to improve the performance of the model

# Step 8: Model deployment
# Save the trained model to a file
model.save("emotion_model.h5")

# Load the saved model in a new Python script for deployment
from tensorflow.keras.models import load_model

deployed_model = load_model("emotion_model.h5")

