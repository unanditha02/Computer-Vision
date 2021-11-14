import numpy as np
import cv2
import matplotlib.pyplot as plt
from opts import get_opts

from helper import plotMatches

#Import necessary functions

from planarH import computeH_ransac
from matchPics import matchPics
from planarH import compositeH

# initializing variables
opts = get_opts()
opts.sigma = 0.4
pano_right = cv2.imread('right1.jpg')
pano_left = cv2.imread('left1.jpg')

# scaling the image dimensions
scale_percent = 40 # percent of original size
width = int(pano_right.shape[1] * scale_percent / 100)
height = int(pano_right.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
pano_right = cv2.resize(pano_right, dim, interpolation = cv2.INTER_AREA)
pano_left = cv2.resize(pano_left, dim, interpolation = cv2.INTER_AREA)

# finding the matches 
matches, locs1, locs2 = matchPics(pano_right, pano_left, opts)
# display matched features
# plotMatches(pano_right, pano_left, matches, locs1, locs2)

# extracting matches
x1 = np.zeros(shape=(len(matches), 2))
x2 = np.zeros(shape=(len(matches), 2))
for i in range(len(matches)):
    x1[i] = locs1[matches[i][0]]
    x2[i] = locs2[matches[i][1]]
x1[:, [0, 1]] = x1[:, [1, 0]]
x2[:, [0, 1]] = x2[:, [1, 0]]

bestH2to1, max_inliers = computeH_ransac(x2, x1, opts)

pano_img = cv2.warpPerspective(pano_right, bestH2to1, (int(1.5*pano_right.shape[1]), int(1.2*pano_right.shape[0])))
pano_img[0:pano_left.shape[0], 0:pano_left.shape[1]] = pano_left
plt.imshow(cv2.cvtColor(pano_img, cv2.COLOR_RGB2BGR))
plt.axis('off')
plt.title("Panorama")
plt.show()
