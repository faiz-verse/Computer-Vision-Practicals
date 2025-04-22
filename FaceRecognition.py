# Face Recognition / ORB Recognition

import cv2
import numpy as np

# Load images in grayscale
img_object = cv2.imread('./Images/modi 1.jpg', cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread('./Images/modi 2.jpg', cv2.IMREAD_GRAYSCALE)

# Check if images loaded properly
if img_object is None or img_scene is None:
    print("One or both images could not be loaded. Check the file paths.")
    exit()

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(img_object, None)
kp2, des2 = orb.detectAndCompute(img_scene, None)

# Check descriptors before proceeding
if des1 is not None and des2 is not None:
    # BFMatcher with Hamming distance (ideal for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Perform knn match
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    MIN_MATCH_COUNT = 15
    if len(good_matches) > MIN_MATCH_COUNT:
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Get the corners of the object image
        h, w = img_object.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # Draw the detected object outline
        img_scene_color = cv2.cvtColor(img_scene, cv2.COLOR_GRAY2BGR)
        cv2.polylines(img_scene_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

        # Optional: draw matches for visualization
        img_matches = cv2.drawMatches(img_object, kp1, img_scene_color, kp2, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Show results
        cv2.imshow("Detected Object", img_scene_color)
        cv2.imshow("Matches", img_matches)

        print(f"Number of matches:  {len(good_matches)}")
    else:
        print(f"Not enough good matches found - {len(good_matches)}")
else:
    print("Descriptors not found in one or both images.")

cv2.waitKey(0)
cv2.destroyAllWindows()
