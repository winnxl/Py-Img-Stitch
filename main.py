# import matplotlib
import cv2
import numpy as np

# Computer soft-key points and descriptors for l/R images
# Compute distances b/w descriptors
# Select best matches for each descriptor
# Run RANSAC to estimate homography
# Warp to align for stitching
# Stitch

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame


print("STARTING!", cv2.__version__)

# img1 = cv2.imread('res/map_1_p/s2.jpg')     # Right
# img2 = cv2.imread('res/map_1_p/s1.jpg')     # Left

img2 = cv2.imread('res/map_1_p/4.png')
img1 = cv2.imread('res/map_1_p/3.png')

# img1 = cv2.imread('res/map_1_p/2.png')      # Right
# img2 = cv2.imread('res/map_1_p/3.png')       # Left

# img1 = cv2.imread('output/m/o-2-3.png')      # Right
# img2 = cv2.imread('output/m/o-2-3_1.png')       # Left


sift = cv2.xfeatures2d.SIFT_create()
key_pts_1, descrip_1 = sift.detectAndCompute(img1, None)
key_pts_2, descrip_2 = sift.detectAndCompute(img2, None)

# cv2.imshow('orig_img1_keypts', cv2.drawKeypoints(img1, key_pts_1, None))
# cv2.imshow('orig_img2.keypts', cv2.drawKeypoints(img2, key_pts_2, None))
# cv2.waitKey(0)  # Wait for a keypress. Change arg to 25 to close after 25 ms automatically.
# cv2.destroyAllWindows()

# FLANN MATCHER
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
match = cv2.FlannBasedMatcher(index_params, search_params)
matches = match.knnMatch(descrip_1, descrip_2, k=2)

# BFMatcher
# match = cv2.BFMatcher()
# matches = match.knnMatch(descrip_1, descrip_2, k=2)

# Find 'GOOD' Matches
good = []
for m,n in matches:
    if m.distance < 0.03*n.distance:
        good.append(m)

# Output Image with Matches
draw_params = dict(matchColor=(0, 255, 0), # draw matches in green color
                   singlePointColor=None,
                   flags=2)

img3 = cv2.drawMatches(img1, key_pts_1, img2, key_pts_2, good, None, **draw_params)
cv2.imshow("original_image_drawMatches.png", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --------------------------
# Find Homography
# Only stitch if > 10 matches
MIN_MATCH_COUNT = 5
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([key_pts_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([key_pts_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    homo_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w, _ = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(pts, homo_matrix)

    # Show where image overlaps
    # cv2.imshow("original_img2.png", cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Warp Image
    # TODO: Is this necessary for our use case? I suppose if screenshots were zoomed in different amounts
    dst = cv2.warpPerspective(img1, homo_matrix, (img2.shape[1] + img1.shape[1], img2.shape[0] + img1.shape[0])) #TODO: Change img1.shape[0] to overlap size
    # dst = cv2.warpPerspective(img1, homo_matrix, (img2.shape[1] + img1.shape[1], img2.shape[0]))


    dst[0:img2.shape[0], 0:img2.shape[1]] = img2
    dst = trim(dst)
    cv2.imshow("stitched.png", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    cv2.imwrite('output/temp.png', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
