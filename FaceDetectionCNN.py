# Face Detection using CNN (Wont work as dlib is not installed)

import dlib
import cv2

# Load CNN face detector
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# Load image
image = cv2.imread("8bit.png")  # ← Replace with your image path
if image is None:
    print("Image not found!")
    exit()

# Resize (optional, for speed)
# image = cv2.resize(image, (640, 480))

# Convert to RGB (dlib needs RGB)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces
faces = cnn_face_detector(rgb_image, 1)  # 1 = upsample once

# Draw rectangles
for face in faces:
    x1 = face.rect.left()
    y1 = face.rect.top()
    x2 = face.rect.right()
    y2 = face.rect.bottom()
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show image
cv2.imshow("CNN Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
