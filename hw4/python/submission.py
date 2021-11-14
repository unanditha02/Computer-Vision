"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import util
import math
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
import random
import submission as sub
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    trans = [[1/M, 0, 0],
			[0, 1/M, 0],
			[0, 0, 1]]
    N = len(pts1)

    x1_norm = pts1/M        # left
    x2_norm = pts2/M        # right
    U = np.zeros(shape=((N, 9)))

    for i in range(N):
        U[i] = [x2_norm[i][0]*x1_norm[i][0], x2_norm[i][0]*x1_norm[i][1], x2_norm[i][0], x2_norm[i][1]*x1_norm[i][0], x2_norm[i][1]*x1_norm[i][1], x2_norm[i][1], x1_norm[i][0], x1_norm[i][1], 1]
    u, w, v = np.linalg.svd(U.T @ U)
		
    index = np.argmin(w)

    F_vec = v[index,:]
    F_norm = F_vec.reshape(3,3)
    # Denormalize
    F_mid = util.refineF(F_norm, x1_norm, x2_norm)
    F = np.transpose(trans) @ F_mid @ trans

    np.savez("q2_1.npz", F=F, M=M)
    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = K2.T @ F @ K1
    np.savez("q3_1.npz", E=E, F=F)
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    N = len(pts1)
    P = np.zeros(shape=((N,4)))
    M1 = C1
    M2 = C2
    for i in range(N):
        u1 = pts1[i, 0]
        v1 = pts1[i, 1]
        u2 = pts2[i, 0]
        v2 = pts2[i, 1]
        A = np.array([[M1[2][0]*u1 - M1[0][0], M1[2][1]*u1 - M1[0][1], M1[2][2]*u1 - M1[0][2], M1[2][3]*u1 - M1[0][3]],
                      [M1[2][0]*v1 - M1[1][0], M1[2][1]*v1 - M1[1][1], M1[2][2]*v1 - M1[1][2], M1[2][3]*v1 - M1[1][3]],
                      [M2[2][0]*u2 - M2[0][0], M2[2][1]*u2 - M2[0][1], M2[2][2]*u2 - M2[0][2], M2[2][3]*u2 - M2[0][3]],
                      [M2[2][0]*v2 - M2[1][0], M2[2][1]*v2 - M2[1][1], M2[2][2]*v2 - M2[1][2], M2[2][3]*v2 - M2[1][3]]])

        u, w, v = np.linalg.svd(A.T @ A)
        index = np.argmin(w)
        P[i,:] = v[index,:]/v[index,3]
   

    pr1 = (C1 @ P.T).T
    pr2 = (C2 @ P.T).T

    pr1 = np.divide(pr1, pr1[:,-1][:,None])
    pr2 = np.divide(pr2, pr2[:,-1][:,None])
    pr1 = pr1[:,0:2]
    pr2 = pr2[:,0:2]

    norm1 = (np.linalg.norm((pts1 - pr1),axis=1))**2
    norm2 = (np.linalg.norm((pts2 - pr2), axis=1))**2
    err = np.sum(norm1+norm2)

    P = P[:,0:3]
    
    return P, err

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    sigma = 2
    v = np.array([x1, y1, 1])
    l = F.dot(v)
    s = np.sqrt(l[0]**2+l[1]**2)
    l = l/s
    size = 27
    w = int(size/2)
    w1 = im1[y1-w:y1+w+1, x1-w:x1+w+1, :]
    for i in range(3):
        w1[:,:,i] = gaussian_filter(w1[:,:,i], sigma=sigma)

    lowest_dist = 9999
    yshape2, xshape2, _ = im2.shape
    for yi in range(w, yshape2-w):
        xi = math.ceil((-l[2] - l[1]*yi) / l[0])
        w2 = im2[yi-w:yi+w+1, xi-w:xi+w+1, :]
        try:
            for i in range(3):
                w2[:,:,i] = gaussian_filter(w2[:,:,i], sigma=sigma)
            dist = np.linalg.norm(w1-w2)

            if(dist<lowest_dist):
                lowest_dist = dist
                x2 = xi
                y2 = yi
        except:
            x2 = 0
            y2 = 0

    return x2, y2
    

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=50, tol=0.42):
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    N = len(pts1)
    bestF = np.zeros(shape=(3, 3))
    random.seed(99)
    
    for i in range(nIters):
        print('i: ',i)
        samples = random.sample(range(1, N), 8)
        j = 0
        inlier_max = 0
        inliers = np.zeros((N,1), dtype=bool)
        pts_sample1 = np.zeros(shape=(8, 2))
        pts_sample2 = np.zeros(shape=(8, 2))
        pts2_pred = np.zeros(shape=(N, 2))
        for s in samples:
            pts_sample1[j] = pts1[s, :]
            pts_sample2[j] = pts2[s, :]
            j += 1
            
        F = sub.eightpoint(pts_sample1, pts_sample2, M)
        for k in range(N):
            pts2_pred[k, 0], pts2_pred[k, 1] = sub.epipolarCorrespondence(im1, im2, F, pts1[k][0].item(), pts1[k][1].item())

        for n in range(N):
            p1 = np.array([pts1[n][0].item(), pts1[n][0].item(), 1])
            p2 = np.array([pts2_pred[n,0], pts1[n,1], 1])

            error = (p2.T) @ F @ p1
            if error < tol:
                inliers[n] = True
            else:
                inliers[n] = False
                
        inlier_count = np.count_nonzero(inliers)
        if inlier_count>inlier_max:
            inlier_max = inlier_count
            bestF = F
            bestInliers = inliers
    
    indices = np.where(bestInliers == True)
    p1final = np.zeros(shape=(len(indices),2))
    p2final = np.zeros(shape=(len(indices),2))
    for z in range(len(indices)):
        p1final[z,:] = pts1[indices[0][z],:]
        p2final[z,:] = pts2[indices[0][z],:]
    F = sub.eightpoint(p1final, p2final, M)
    return F, bestInliers

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    theta = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    r0 = r / theta
    K = [[0, -r0[2], r0[1]], [r0[2], 0, -r0[0]], [-r0[1], r0[0], 0]]
    R = np.eye(3) + (math.sin(theta)*K) + ((1 - math.cos(theta))*(K**2))

    return R


'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    theta = math.acos((np.trace(R) - 1)/2)
    omega = (1 / 2*math.sin(theta))* [[R[2][1] - R[1][2]][R[0][2] - R[2][0]][R[1][0] - R[0][1]]]
    r = theta * omega

    return r

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
