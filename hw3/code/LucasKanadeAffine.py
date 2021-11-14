import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """
    i = 0
    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    p = np.zeros(6)
    while True: 
        # warp It1 - find I(W(It1))
        rect = [0, 0, It.shape[0]-1, It.shape[1]-1]
        tempInt = RectBivariateSpline(np.linspace(0, It.shape[0]-1, num=It.shape[0]), np.linspace(0, It.shape[1]-1, num=It.shape[1]), It)
        It1Int = RectBivariateSpline(np.linspace(0, It1.shape[0]-1, num=It1.shape[0]), np.linspace(0, It1.shape[1]-1,num=It.shape[1]), It1)
        xi, yi = np.mgrid[rect[0]:rect[2]+1, rect[1]:rect[3]+1]
        M = np.array([[p[0]+1, p[1], p[2]], [p[3], p[4]+1, p[5]], [0.0, 0.0, 1.0]])
        # M = np.array([[p[0]+1, p[1], p[2]], [p[3], p[4]+1, p[5]], [0.0, 0.0, 1.0]])
        x_warp = M[0][0] * xi + M[0][1] * yi + M[0][2]
        y_warp = M[1][0] * xi + M[1][1] * yi + M[1][2]

        # yWarpShape, xWarpShape = warped1.shape
        commonPts = ((x_warp>=0) & (x_warp<=It.shape[0]) & (y_warp>=0) & (y_warp<=It.shape[1]))
        x_warp_common = x_warp[commonPts]
        y_warp_common = y_warp[commonPts]
        xi_common = xi[commonPts]
        yi_common = yi[commonPts]
        tempInt = tempInt.ev(yi_common, xi_common).T
        warped1 = It1Int.ev(y_warp_common, x_warp_common).T 

        error = (tempInt - warped1)
        # gradient 
        gradientX = It1Int.ev(y_warp_common, x_warp_common, dx = 0, dy = 1).T
        gradientY = It1Int.ev(y_warp_common, x_warp_common, dx = 1, dy = 0).T 
        grad = np.vstack((gradientX.flatten(), gradientY.flatten())).T

        xi = xi.flatten()
        yi = yi.flatten()
        gradientX = gradientX.flatten()
        gradientY = gradientY.flatten()

        # jacobian =  np.array([[xi, yi, np.ones(xi.shape), np.zeros(xi.shape), np.zeros(xi.shape), np.zeros(xi.shape)],
        # [np.zeros(xi.shape), np.zeros(xi.shape), np.zeros(xi.shape), xi, yi, np.ones(xi.shape)]])
        # steepestDesc = np.tensordot(grad, jacobian)
        steepestDesc = np.vstack((gradientX*xi_common, gradientX*yi_common, gradientX, gradientY*xi_common, gradientY*yi_common, gradientY)).T

        hessian = np.dot(steepestDesc.T,  steepestDesc)
        sDxError = np.dot(np.transpose(steepestDesc), error.flatten())
        delta_p = np.linalg.inv(hessian) @ sDxError
        p += delta_p
        i +=1

        if (np.linalg.norm(delta_p) <= threshold or i>num_iters):
            break
    
    return M
