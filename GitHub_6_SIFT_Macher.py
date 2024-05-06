import cv2
import matplotlib.pyplot as plt

img1_path = r''  # Path to the first image
img2_path = r''  # Path to the second image

def find_and_match_features(img1_path, img2_path, d=0.5, crop_size=64):
    img1 = cv2.imread(img1_path)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(img2_path)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    matched_points1 = []
    matched_points2 = []
    for m, n in matches:
        if m.distance < d * n.distance:
            pt = keypoints2[m.trainIdx].pt
            if (crop_size // 2 <= pt[0] < img2.shape[1] - crop_size // 2) and \
               (crop_size // 2 <= pt[1] < img2.shape[0] - crop_size // 2):
                good_matches.append(m)
                matched_points1.append(keypoints1[m.queryIdx].pt)
                matched_points2.append(pt)

    if not matched_points2:
        print("No suitable matches found that allow a full 64x64 crop.")
        return None, None, None, None  # Return None if no suitable matches were found.

    # Crop around the first suitable match in the second image
    x, y = int(matched_points2[0][0]), int(matched_points2[0][1])
    print(f"First suitable match found at ({x}, {y})")
    x0, y0 = x - crop_size // 2, y - crop_size // 2
    x1, y1 = x0 + crop_size, y0 + crop_size
    crop_img = img2[y0:y1, x0:x1]

    # Visualize the cropped region
    plt.figure(figsize=(3, 3))
    plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    plt.title('Cropped 64x64 Region Around First Suitable Match')
    plt.show()

    # Visualize matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.imshow(img_matches)
    plt.title('SIFT Feature Matches')
    plt.show()

    return keypoints1, keypoints2, good_matches, crop_img