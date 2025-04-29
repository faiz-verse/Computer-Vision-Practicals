import numpy as np
import cv2
import os
import glob

# Parameters
board_size = (8, 6)  # Inner corners (NOT number of squares)
square_size = 50  # In arbitrary units (e.g., millimeters)
num_images = 15
output_dir = "../Images/checkboard"

squares_x = board_size[0] + 1  # Number of squares horizontally
squares_y = board_size[1] + 1  # Number of squares vertically

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# STEP 1: Generate perfect chessboard images
for i in range(num_images):
    img = np.ones((squares_y * square_size, squares_x * square_size), np.uint8) * 255

    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                cv2.rectangle(
                    img,
                    (x * square_size, y * square_size),
                    ((x + 1) * square_size, (y + 1) * square_size),
                    0,
                    -1
                )

    filename = os.path.join(output_dir, f"calib_{i:02d}.png")
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(filename, img_blurred)

print(f"✅ Generated {num_images} chessboard images in '{output_dir}'.")

# STEP 2: Prepare object points
objp = np.zeros((board_size[1] * board_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

# Termination criteria for subpixel corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# STEP 3: Detect corners
images = glob.glob(os.path.join(output_dir, "*.png"))
collage_images = []

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, board_size, None)

    if ret:
        objpoints.append(objp)

        # Refine corner locations to subpixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and collect images for collage
        img_drawn = cv2.drawChessboardCorners(img, board_size, corners2, ret)
        collage_images.append(cv2.resize(img_drawn, (400, 300)))

if not collage_images:
    raise ValueError("❌ No corners were detected. Calibration failed.")

# STEP 4: Calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n🎯 Calibration Results:")
print(f"Camera Matrix:\n{camera_matrix}")
print(f"Distortion Coefficients:\n{dist_coeffs.ravel()}")

# STEP 5: Create collage for visualization
cols = 5
rows = (len(collage_images) + cols - 1) // cols

# Fill missing slots with white images if necessary
white = np.ones_like(collage_images[0]) * 255
while len(collage_images) < rows * cols:
    collage_images.append(white.astype(np.uint8))

# Stack images
rows_img = [np.hstack(collage_images[i * cols:(i + 1) * cols]) for i in range(rows)]
collage = np.vstack(rows_img)

# Display final collage
cv2.imshow("All Detected Chessboards", collage)
cv2.waitKey(0)
cv2.destroyAllWindows()
