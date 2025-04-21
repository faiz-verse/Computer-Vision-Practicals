# ❌ AAH! Camera Calibration NOT WORKING

import cv2
import numpy as np
import glob
import os

# Checkerboard parameters
CHECKERBOARD = (9, 6)  # (width, height) of the checkerboard squares

# Prepare object points based on the checkerboard size
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Correct path to the folder containing checkerboard images
image_dir = "./Images/checkboard"  # Replace this with the correct path to the folder

# Search for all .jpg or .tif images in the folder
images = glob.glob(os.path.join(image_dir, "*.jpg"))  # Or use "*.jpg" if your images are in JPG format

# Check if images are loaded
if len(images) == 0:
    print("No images found in the directory. Please check the path.")
    exit()

# Initialize gray variable outside the loop to hold image dimensions
gray = None

# Loop through each image to find the corners
for fname in images:
    img = cv2.imread(fname)

    if img is None:
        print(f"Error loading image: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)  # Add object points
        imgpoints.append(corners)  # Add image points

        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)  # Wait for 500 ms
    else:
        print(f"Checkerboard not detected in image: {fname}")
        # Show the image for debugging purposes
        cv2.imshow("Failed to Detect", img)
        cv2.waitKey(500)  # Show image for 500ms

# Ensure gray is not None
if gray is None:
    print("No valid images found. Exiting...")
    exit()

# Camera calibration
if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("Camera Calibration successful!")
        print("Camera Matrix: \n", mtx)
        print("Distortion Coefficients: \n", dist)

        # Save the calibration parameters
        try:
            np.savez("calibration_params.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            print("Calibration parameters saved successfully.")
        except Exception as e:
            print(f"Error saving calibration parameters: {e}")

    else:
        print("Camera Calibration failed.")
else:
    print("No valid object points or image points found. Exiting...")

cv2.destroyAllWindows()
