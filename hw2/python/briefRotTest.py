import numpy as np
import cv2
from numpy.lib.function_base import append
from matchPics import matchPics
import scipy
import matplotlib.pyplot as plt
from opts import get_opts

opts = get_opts()
#Q2.1.6
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')
num_of_matches = []
rotations = []
for i in range(36):
	#Rotate Image
	rotations.append(i*10)
	rot_cv_cover = scipy.ndimage.rotate(cv_cover, rotations[i], reshape=True)

	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(cv_cover, rot_cv_cover, opts)

	#Update histogram
	num_of_matches.append(len(matches))

#Display histogram

print(num_of_matches, rotations)
plt.xlabel("Rotation (in degrees)")
plt.ylabel("Number of Matches")
plt.title("Matches")
plt.bar(rotations, num_of_matches, width=10)
plt.show()