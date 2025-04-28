# Image stitching

import cv2
import numpy as np

# Load images
image = cv2.imread('./Images/image-stitching-1.jpg')
image2 = cv2.imread('./Images/image-stitching-2.jpg')


# helper function to show images by resizing
def show_resized(window_name, img, scale=0.4):
    small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    cv2.imshow(window_name, small)


# Check if images are loaded correctly
if image is None or image2 is None:
    print("Error: Could not load one or both images.")
else:
    show_resized("image 1", image)
    show_resized("image 2", image2)

    # Resize image2 to match image1 dimensions
    image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(gray, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Draw keypoints on the first image
    img_kp1 = cv2.drawKeypoints(gray, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_resized("Key points in image 1", img_kp1)  # Show image with keypoints

    # Draw keypoints on the first image
    img_kp2 = cv2.drawKeypoints(gray2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_resized("Key points in image 2", img_kp2)  # Show image with keypoints

    # FLANN-based matcher parameters
    index_params = dict(algorithm=0, trees=5)  # KD-Tree algorithm
    search_params = dict()  # Default search params

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors using KNN
    matches = flann.knnMatch(des1, des2, k=2)  # Match des1 (image1) with des2 (image2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Lowe’s ratio test
            good_matches.append(m)

    # Draw matches between the two images
    match_img = cv2.drawMatches(image, kp1, image2, kp2, good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    show_resized("Match image", match_img)  # Show the matched keypoints

    print(f"Number of good matches: {len(good_matches)}")

    # If enough good matches are found, compute homography
    if len(good_matches) > 7:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp the first image to align with the second
        h, w = gray.shape
        aligned_img = cv2.warpPerspective(image, M, (w, h))

        # Create a blank canvas for result (side-by-side display)
        result = np.zeros((h, w * 2, 3), dtype=np.uint8)

        # Place original image2 on the left and aligned image on the right
        result[0:h, 0:w] = image2
        result[0:h, w:w * 2] = aligned_img

        show_resized("Final stiched Image", result)  # Show aligned image

    else:
        print("Not enough matches found for homography.")

cv2.waitKey(0)
cv2.destroyAllWindows()
