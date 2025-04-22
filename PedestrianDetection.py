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
