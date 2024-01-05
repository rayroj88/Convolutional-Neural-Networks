import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def train_face_model(faces, nonfaces):
    """
    Trains a convolutional neural network (CNN) model to classify images as faces or non-faces.

    Args:
    - faces: A numpy array of shape (n_faces, height, width) containing grayscale images of faces.
    - nonfaces: A numpy array of shape (n_nonfaces, height, width) containing grayscale images of non-faces.

    Returns:
    - model: A trained Keras CNN model.
    - history: A Keras history object containing information about the training process.
    """
    
    # Reshape the inputs to include the channel dimension (grayscale => 1 channel)
    faces = faces.reshape((faces.shape[0], faces.shape[1], faces.shape[2], 1))  # Line 26: Make sure this line
                                                                                # is indented consistently with the surrounding code
    nonfaces = nonfaces.reshape((nonfaces.shape[0], nonfaces.shape[1], nonfaces.shape[2], 1))

    # Normalize values
    faces = faces.astype('float32') / 255.0
    nonfaces = nonfaces.astype('float32') / 255.0

    # Assign labels for faces and nonfaces
    face_labels = np.ones((faces.shape[0], 1))
    nonface_labels = np.zeros((nonfaces.shape[0], 1))

    # Stack faces and nonfaces
    data = np.vstack((faces, nonfaces))
    labels = np.vstack((face_labels, nonface_labels))

    # Shuffle the data and labels
    data, labels = shuffle(data, labels, random_state=42)

    # Define the CNN model architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(faces.shape[1], faces.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(data, labels, epochs=10, batch_size=64, validation_split=0.3)

    return model, history
