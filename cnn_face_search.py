import numpy as np
import cv2


def cnn_face_search(image, model, face_size, scale, result_number):
    """
    Searches for faces in an image using a convolutional neural network (CNN) model.

    Args:
        image (numpy.ndarray): The input image as a numpy array.
        model (tensorflow.keras.models.Sequential): The CNN model to use for face detection.
        face_size (tuple): A tuple containing the height and width of the face window to use for detection.
        scale (float): The scaling factor to use for the image. The scale is measured with the respect to 
                       the original image face size. For example, if the original image face size is (31, 25) 
                       and the scale is 2.0, then the scaled image face size will be (62, 50).
        result_number (int): The maximum number of results to return.

    Returns:
        tuple: A tuple containing two elements:
            - A list of tuples, where each tuple contains the confidence score, center row and column coordinates 
              of the detected face, and the top, bottom, left, and right coordinates of the bounding box around the face.
            - A numpy array containing the confidence scores for each pixel in the input image.
    """
    
	# YOUR CODE HERE
 
    # Ensure the pixel values are normalized
    image_scaled = cv2.resize(image, (0, 0), fx=1.0/scale, fy=1.0/scale).astype(np.float32) / 255.0
    scaled_face_size = (int(face_size[0] * scale), int(face_size[1] * scale))

    # Prepare the window
    windows = []
    for y in range(0, image_scaled.shape[0] - scaled_face_size[0], scaled_face_size[0] // 2):
        for x in range(0, image_scaled.shape[1] - scaled_face_size[1], scaled_face_size[1] // 2):
            window = image_scaled[y:y+scaled_face_size[0], x:x+scaled_face_size[1]]
            window_resized = cv2.resize(window, (face_size[1], face_size[0])) # Ensure the order of face_size is (width, height)
            window_reshaped = window_resized.reshape(face_size[0], face_size[1], 1) # Add a channel dimension
            windows.append(window_reshaped)

    # Create a numpy array of windows
    windows_arr = np.array(windows)

    # Predict using the CNN model
    prediction = model.predict(windows_arr)

    # Associate each window with its location in the original image
    results = []
    index = 0
    for y in range(0, image_scaled.shape[0] - scaled_face_size[0], scaled_face_size[0] // 2):
        for x in range(0, image_scaled.shape[1] - scaled_face_size[1], scaled_face_size[1] // 2):
            confidence = prediction[index][0]
            if confidence > 0.5:  # Only consider windows with a confidence score above 0.5
                center_row = y * scale
                center_col = x * scale
                top = max(center_row - scaled_face_size[0] // 2, 0)
                bottom = min(center_row + scaled_face_size[0] // 2, image.shape[0])
                left = max(center_col - scaled_face_size[1] // 2, 0)
                right = min(center_col + scaled_face_size[1] // 2, image.shape[1])
                results.append((confidence, center_row, center_col, top, bottom, left, right))
            index += 1

    # Sort results by confidence score in descending order and select top results
    results = sorted(results, key=lambda x: x[0], reverse=True)[:result_number]

    # Generate confidence scores array for the entire image
    scores = np.zeros_like(image_scaled)
    for result in results:
        (_, center_row, center_col, _, _, _, _) = result
        scores[int(center_row/scale), int(center_col/scale)] = 1  # Mark the center of the detected face

    return results, scores
