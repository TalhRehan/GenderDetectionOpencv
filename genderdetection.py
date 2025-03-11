from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

# Load pre-trained model
model = load_model('gender_detection_model_byTalha.h5')

# Start video capture
webcam = cv2.VideoCapture(0)

# Correct class labels (man and woman)
classes = ['man', 'woman']  # Check the exact labels the model was trained on

while webcam.isOpened():
    # Read frame from webcam
    status, frame = webcam.read()

    # Apply face detection
    face, confidence = cv.detect_face(frame)

    # Loop through detected faces
    for idx, f in enumerate(face):
        # Get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw placeholder rectangle over face
        color = (0, 255, 0)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocess the cropped face for model prediction
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Predict gender
        conf = model.predict(face_crop)[0]

        # Get label with max confidence
        idx = np.argmax(conf)
        label = classes[idx]

        # Debug: Print label to confirm correct classification
        print(f"Predicted label: {label}, Confidence: {conf[idx] * 100:.2f}%")

        # Set label and color based on gender
        if label == "man":
            label = "male"
            color = (255, 0, 0)  # Blue for male
        elif label == "woman":
            label = "female"
            color = (255, 105, 180)  # Pink for female

        # Format label with confidence
        label_text = "{}: {:.2f}%".format(label, conf[idx] * 100)

        # Position label text
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Write label and confidence above face rectangle
        cv2.putText(frame, label_text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        # Draw rectangle with the gender-specific color
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Display output
    cv2.imshow("Gender Detection", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
