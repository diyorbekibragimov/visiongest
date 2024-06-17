import keras
import cv2  # Install opencv-python
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
# Load the TFSMLayer
tfsmlayer = keras.layers.TFSMLayer("Model/model.savedmodel", call_endpoint='serving_default')

# Create a Keras model that includes the TFSMLayer
model = keras.Sequential([
    keras.Input(shape=(224, 224, 3)),  # Ensure input shape matches the expected input shape for the TFSMLayer
    tfsmlayer
])

# Load the labels
class_names = open("Model/labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 224

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y:y + h, x:x + w]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

            imgWhite_reshaped = np.expand_dims(imgWhite, axis=0)
            
            prediction = model.predict(imgWhite_reshaped)
            prediction_array = prediction['sequential_3']

            # Find the index of the highest probability
            index = np.argmax(prediction_array)

            # Extract the class name using the index
            class_name = class_names[index].strip()  # Assuming class_names is a list of class names

            # Extract the confidence score for the predicted class
            confidence_score = prediction_array[0, index]
            cv2.putText(img, class_name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

            print(f"Predicted class: {class_name}, Confidence: {confidence_score:.4f}")
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

            imgWhite_reshaped = np.expand_dims(imgWhite, axis=0)
            
            prediction = model.predict(imgWhite_reshaped)
            prediction_array = prediction['sequential_3']

            # Find the index of the highest probability
            index = np.argmax(prediction_array)

            # Extract the class name using the index
            class_name = class_names[index].strip()  # Assuming class_names is a list of class names

            # Extract the confidence score for the predicted class
            confidence_score = prediction_array[0, index]

            cv2.putText(img, class_name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

            print(f"Predicted class: {class_name}, Confidence: {confidence_score:.4f}")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Show the image in a window
    cv2.imshow("Webcam Image", image)
    image_resized = cv2.resize(image, (224, 224))  # Resize the image to 224x224

    # Now, convert the resized image to a numpy array and reshape it for the model
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image_array / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    prediction_array = prediction['sequential_3']

    # Find the index of the highest probability
    index = np.argmax(prediction_array)

    # Extract the class name using the index
    class_name = class_names[index].strip()  # Assuming class_names is a list of class names

    # Extract the confidence score for the predicted class
    confidence_score = prediction_array[0, index]

    print(f"Predicted class: {class_name}, Confidence: {confidence_score:.4f}")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()