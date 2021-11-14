import numpy as np
import cv2
import math
import random

def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	N = len(x1)
	A = np.zeros(shape=((N * 2, 9)))
	j = 0
	for i in range(N):
		A[j] = np.array([x2[i][0], x2[i][1], 1, 0, 0, 0, -x2[i][0]*x1[i][0], -x2[i][1]*x1[i][0], -x1[i][0]])
		j += 1
		A[j] = np.array([0, 0, 0, x2[i][0], x2[i][1], 1, -x2[i][0]*x1[i][1], -x2[i][1]*x1[i][1], -x1[i][1]])
		j += 1

	w, v = np.linalg.eig(np.transpose(A) @ A)
	index = np.argmin(w)
	H = v[:, index]
	H2to1 = H.reshape(3,3)
	return H2to1


def computeH_norm(x1, x2):
	N = len(x1)
	#Q2.2.2
	#Compute the centroid of the points
	x1_mean_x = np.mean(x1[:, 0])
	x1_mean_y = np.mean(x1[:, 1])
	x2_mean_x = np.mean(x2[:, 0])
	x2_mean_y = np.mean(x2[:, 1])

	#Shift the origin of the points to the centroid
	translate1 = [[1, 0, -x1_mean_x],
			[0, 1, -x1_mean_y],
			[0, 0, 1]]
	translate2 = [[1, 0, -x2_mean_x],
			[0, 1, -x2_mean_y],
			[0, 0, 1]]

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	dist1 = np.zeros(shape=(N, 1))
	dist2 = np.zeros(shape=(N, 1))

	dist1 = np.sqrt(np.square(x1[:, 0] - x1_mean_x) + np.square(x1[:, 1] - x1_mean_y))
	dist2 = np.sqrt(np.square(x2[:, 0] - x2_mean_x) + np.square(x2[:, 1] - x2_mean_y))
	
	d1_scale = math.sqrt(2) / max(dist1)
	d2_scale = math.sqrt(2) / max(dist2)

	scale1 = [[d1_scale, 0, 0],
			[0, d1_scale, 0],
			[0, 0, 1]]
	scale2 = [[d2_scale, 0, 0],
			[0, d2_scale, 0],
			[0, 0, 1]]

	#Similarity transform 1
	T1 = np.matmul(translate1, scale1)
	T2 = np.matmul(translate2, scale2)
	x1 = np.append(x1, np.ones(shape=(N,1)), axis=1)
	x2 = np.append(x2, np.ones(shape=(N,1)), axis=1)
	
	#Similarity transform 2
	x1_norm = np.transpose(np.matmul(T1, np.transpose(x1)))
	x2_norm = np.transpose(np.matmul(T2, np.transpose(x2)))
	
	# Compute homography
	H = computeH(x1_norm, x2_norm)

	#Denormalization
	H_mid = np.matmul(H, T2)
	H2to1 = np.matmul(np.linalg.inv(T1), H_mid)

	return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
	N = len(locs1)
	random.seed(99)
	locs2 = np.append(locs2, np.ones(shape=(N,1)), axis=1)
	bestH2to1 = np.zeros(shape=(3, 3))
	inliers = 0
	for i in range(max_iters):
		samples = random.sample(range(1, N), 4)

		j = 0
		x1 = np.zeros(shape=(4, 2))
		x2 = np.zeros(shape=(4, 2))
		
		for s in samples:
			x1[j] = locs1[s, 0:2]
			x2[j] = locs2[s, 0:2]
			j += 1

		H = computeH_norm(x1, x2)

		inlier_count = 0		
		locs1_pred = np.transpose((H @ np.transpose(locs2)))

		for n in range(N):
			locs1_pred[n, :] = locs1_pred[n, :] / locs1_pred[n, 2]
			error = np.linalg.norm(locs1_pred[n, 0:2] - locs1[n])
			if error < inlier_tol:
				inlier_count += 1

		if inlier_count > inliers:
			inliers = inlier_count
			bestH2to1 = H


	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	warped_template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))
	mask = np.ones(shape=template.shape, dtype="uint8") * 255
	warped_mask = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0]))
	warped_mask_inv = cv2.bitwise_not(warped_mask)
	desk_masked = cv2.bitwise_and(img, img, mask=warped_mask_inv[:, :, 0])
	

	composite_img = cv2.add(warped_template, desk_masked)
	#Create mask of same size as template

	#Warp mask by appropriate homography
	# warped_mask = H2to1 @ mask
	#Warp template by appropriate homography
	# warped_template = H2to1 @ template
	#Use mask to combine the warped template and the image
	
	return composite_img


