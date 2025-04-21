# Face Detection

# Using HOG (works in google collab as cant install the dlib package)

import cv2
import dlib
from google.colab.patches import cv2_imshow # Import cv2_imshow

# Load HOG face detector from dlib
hog_face_detector = dlib.get_frontal_face_detector()

# Load the image
image_path = "/content/8bit.png"  # Provide the correct image path
image = cv2.imread(image_path)

# Check if image is loaded successfully
if image is None:
    print("Error: Image not found!")
else:
    # Convert the image to grayscale (BGR -> Grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = hog_face_detector(gray)

    # Print the number of faces detected
    print("Number of faces detected:", len(faces))

    # Draw bounding boxes around detected faces
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

    # Display the modified image with bounding boxes
    cv2_imshow(image)  # Use cv2_imshow instead of cv2.imshow
    cv2.waitKey(1) #changed from 0 to 1
    cv2.destroyAllWindows()  # Close the image window