import cv2
import numpy as np

# Load reference image (known person)
reference_img = cv2.imread('./Images/people_walk.PNG', cv2.IMREAD_GRAYSCALE)

# Check if reference image loaded properly
if reference_img is None:
    print("❌ Failed to load reference image.")
    exit()

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for the reference image
kp1, des1 = sift.detectAndCompute(reference_img, None)
if des1 is None:
    print("❌ No descriptors found in reference image.")
    exit()

# Create BFMatcher
bf = cv2.BFMatcher()

# Open the video file
video_path = './Images/people_walk.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Failed to open video.")
    exit()

frame_count = 0
detected_frames = 0

print("🔍 Scanning video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors for the frame
    kp2, des2 = sift.detectAndCompute(gray_frame, None)

    if des2 is not None and len(des2) > 0:
        # Match descriptors
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        print(f"🔎 Frame {frame_count}: {len(good_matches)} good matches")

        if len(good_matches) >= 10:
            # Extract matched keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if mask is not None:
                inliers = mask.ravel().tolist()
                num_inliers = np.sum(inliers)

                # Calculate match percentage
                match_percentage = (num_inliers / len(good_matches)) * 100

                if match_percentage >= 50:
                    detected_frames += 1
                    print(f"✅ Detected in frame {frame_count} with {num_inliers} inliers ({match_percentage:.2f}% match)")

                    # Draw inlier matches
                    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inliers[i]]
                    match_img = cv2.drawMatches(reference_img, kp1, gray_frame, kp2, inlier_matches, None, flags=2)

                    # Show result in PyCharm
                    cv2.imshow(f"Detected Frame {frame_count}", match_img)
                    cv2.waitKey(1)  # Show each detection quickly
                else:
                    print(f"❌ Frame {frame_count}: Only {match_percentage:.2f}% match")
    else:
        print(f"⚠️ Frame {frame_count}: No descriptors found")

cap.release()
cv2.destroyAllWindows()
print(f"\n🎬 Finished scanning video. Person detected in {detected_frames} frame(s).")
