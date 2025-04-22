import cv2
import numpy as np


def load_images(object_path, scene_path):
    img_object = cv2.imread(object_path, cv2.IMREAD_GRAYSCALE)
    img_scene = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)

    if img_object is None or img_scene is None:
        raise FileNotFoundError("Error loading images!")

    return img_object, img_scene


def detect_keypoints_and_descriptors(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def match_descriptors(des1, des2):
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return good_matches


def find_homography_and_draw(img_object, img_scene, kp1, kp2, good_matches, min_match_count=10):
    if len(good_matches) >= min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = img_object.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img_result = cv2.polylines(cv2.cvtColor(img_scene, cv2.COLOR_GRAY2BGR), [np.int32(dst)], True, (0, 255, 0), 3,
                                   cv2.LINE_AA)
        mask_list = mask.ravel().tolist()
    else:
        print(f"Not enough matches found ({len(good_matches)}).")
        img_result = cv2.cvtColor(img_scene, cv2.COLOR_GRAY2BGR)
        mask_list = None

    return img_result, mask_list


def draw_matches(img_object, kp1, img_scene, kp2, good_matches, mask_list=None):
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=mask_list,
                       flags=2)
    return cv2.drawMatches(img_object, kp1, img_scene, kp2, good_matches, None, **draw_params)


# Implementation
object_path = './Images/checkboard/luffy_single.jpg'
scene_path = './Images/checkboard/luffy_group.jpg'

img_object, img_scene = load_images(object_path, scene_path)
kp1, des1 = detect_keypoints_and_descriptors(img_object)
kp2, des2 = detect_keypoints_and_descriptors(img_scene)

good_matches = match_descriptors(des1, des2)
img_result, mask_list = find_homography_and_draw(img_object, img_scene, kp1, kp2, good_matches)
img_matches = draw_matches(img_object, kp1, img_scene, kp2, good_matches, mask_list)

# Display results
cv2.imshow('Matches', img_matches)
cv2.imshow('Detected Object Region', img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

