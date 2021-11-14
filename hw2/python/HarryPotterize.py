import numpy as np
import cv2
import skimage.color
from opts import get_opts
import matplotlib.pyplot as plt
#Import necessary functions

from planarH import computeH_ransac
from planarH import compositeH
from matchPics import matchPics
from helper import plotMatches

# initializing variables
opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')
matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)

#display matched features
plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

# extracting matches
x1 = np.zeros(shape=(len(matches), 2))
x2 = np.zeros(shape=(len(matches), 2))
for i in range(len(matches)):
    x1[i] = locs1[matches[i][0]]
    x2[i] = locs2[matches[i][1]]
x1[:, [0, 1]] = x1[:, [1, 0]]
x2[:, [0, 1]] = x2[:, [1, 0]]


try:
# compute homography
    bestH2to1, max_inliers = computeH_ransac(x2, x1, opts)
    hp_cover = cv2.resize(hp_cover,(cv_cover.shape[1], cv_cover.shape[0]))
    # finding the composite image
    composite_img = compositeH(bestH2to1, hp_cover, cv_desk)


except:
    composite_img = cv_desk    

plt.imshow(cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
