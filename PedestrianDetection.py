# Pedestrian Detection (1)

import cv2

# Initialize HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load the image
image = cv2.imread('./Images/pedestrians.jpg')  # Replace with your actual image path

# Check if the image is loaded correctly
if image is None:
    print("Error loading image!")
    exit()

# Resize image (optional for better detection)
image = cv2.resize(image, (640, 480))

# Detect people
(rects, weights) = hog.detectMultiScale(
    image,
    winStride=(4, 4),
    padding=(8, 8),
    scale=1.05
)

# Draw bounding boxes
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the result in a window
cv2.imshow("HOG People Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Pedestrian detection (2) with non-maximum supression and labels
import cv2
import numpy as np

# Initialize HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load and resize image
image = cv2.imread('./Images/pedestrians.jpg')
image = cv2.resize(image, (640, 480))

# Detect people
(rects, weights) = hog.detectMultiScale(
    image,
    winStride=(4, 4),
    padding=(8, 8),
    scale=1.05
)

# Threshold for strong detections
threshold = 0.6
strong_rects = [rects[i] for i in range(len(rects)) if weights[i] > threshold]
strong_scores = [float(weights[i]) for i in range(len(rects)) if weights[i] > threshold]

# Apply Non-Maximum Suppression
indices = cv2.dnn.NMSBoxes(
    bboxes=strong_rects,  # Already in (x, y, w, h) format
    scores=strong_scores,
    score_threshold=threshold,
    nms_threshold=0.4
)

# Draw final bounding boxes
for i in indices:
    i = i[0] if isinstance(i, (np.ndarray, list)) else i
    (x, y, w, h) = strong_rects[i]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"{strong_scores[i]:.2f}"
    cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the result
cv2.imshow("Detection result (labels are SVM confidence scores so they are not limited to 0-1)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
