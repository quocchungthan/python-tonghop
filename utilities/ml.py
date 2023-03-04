from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import utilities.metadata as cbtmeta

def initModel():
    model = Sequential()
    # Add a convolutional layer with 32 filters, a 3x3 kernel, and 'relu' activation
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    # Add a max pooling layer with a pool size of 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add a second convolutional layer with 64 filters and a 3x3 kernel
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # Add another max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten the output from the convolutional layers
    model.add(Flatten())
    # Add a fully connected (Dense) layer with 128 neurons and 'relu' activation
    model.add(Dense(128, activation='relu'))
    # Add a dropout layer to prevent overfitting
    model.add(Dropout(0.5))
    # Add the output layer with 6 neurons (one for each emotion) and 'softmax' activation
    model.add(Dense(len(cbtmeta.targets), activation='softmax'))
    # Compile the model with 'categorical_crossentropy' loss and 'adam' optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model