import os
import sys

# Get the absolute path of the script's directory
current_directory = os.path.abspath(os.path.dirname(__file__))
# Get the parent directory
parent_directory = os.path.dirname(current_directory)
# Add the parent directory to sys.path
sys.path.append(parent_directory)

import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from train_face_model import train_face_model
from cnn_face_search import cnn_face_search
from draw_rectangle import draw_rectangle

def main():
    ############### Load data ###############
    output_dir = os.path.join(current_directory, 'output')
    faces_file_path = os.path.join(current_directory, 'data', 'faces1000.npy')
    nonfaces_file_path = os.path.join(current_directory, 'data', 'nonfaces1000.npy')

    # Load numpy files
    faces = np.load(faces_file_path)
    nonfaces = np.load(nonfaces_file_path)
    
    train_faces = faces[:700]
    test_faces = faces[700:]
    train_nonfaces = nonfaces[:700]
    test_nonfaces = nonfaces[700:]

    ############### Train the model ###############
    model, history = train_face_model(train_faces, train_nonfaces)

    # Save the model
    model.save(os.path.join(output_dir, 'face_model.keras'))

    # Plot the training history
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.show()

    ############### Evaluate the model ###############
    X_test = np.concatenate((test_faces, test_nonfaces), axis=0)
    X_test = cv2.normalize(X_test, None, 0, 1, cv2.NORM_MINMAX)

    faces_labels = np.ones((test_faces.shape[0], 1))
    nonfaces_labels = np.zeros((test_nonfaces.shape[0], 1))
    y_test = np.concatenate((faces_labels, nonfaces_labels), axis=0)
    X_test = np.expand_dims(X_test, axis=3)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    print('Test accuracy:', test_acc)

    # Save the test accuracy
    with open(os.path.join(output_dir, 'test_accuracy.txt'), 'w') as f:
        f.write(str(test_acc))


    ############### Detect faces in vjm.bmp ###############
    # Load the image data/vjm.bmp
    img = cv2.imread('data/vjm.bmp', cv2.IMREAD_GRAYSCALE)

    # Normalize image to [0, 1]
    img = img.astype('float32')
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)

    # Load the model
    model = keras.models.load_model(os.path.join(output_dir, 'face_model.keras'))

    # Detect faces
    face_size = (31, 25)
    scale = 1.0
    result_number = 3

    # Time the function
    start_time = time.time()
    results, scores = cnn_face_search(img, model, face_size, scale, result_number)
    end_time = time.time()
    print('Time elapsed:', end_time - start_time)

    # Scale image back to [0, 255]
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Draw bounding boxes
    for result in results:
        (max_val, best_row, best_col, top, bottom, left, right) = result
        cv2.rectangle(img, (left, top), (right, bottom), 255, 2)

    # Save the image
    cv2.imwrite(os.path.join(output_dir, 'vjm_result.png'), img)


    ############### Detect faces in faces.bmp ###############
    # Load the image data/faces.bmp
    img = cv2.imread('data/faces.bmp', cv2.IMREAD_GRAYSCALE)

    # Normalize image to [0, 1]
    img = img.astype('float32')
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)

    # Load the model
    model = keras.models.load_model(os.path.join(output_dir, 'face_model.keras'))

    # Detect faces
    face_size = (31, 25)
    scale = 2.0
    result_number = 2
    results, scores = cnn_face_search(img, model, face_size, scale, result_number)

    # Scale image back to [0, 255]
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Draw bounding boxes
    for result in results:
        (max_val, best_row, best_col, top, bottom, left, right) = result
        cv2.rectangle(img, (left, top), (right, bottom), 255, 2)

    # Save the image
    cv2.imwrite(os.path.join(output_dir, 'faces_result.png'), img)


if __name__ == '__main__':
    main()