import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
   
    i = 0
    
    p = np.zeros(6)
    M = np.array([[1+p[0], p[1], p[2]], [p[3], 1+p[4], p[5]], [0.0, 0.0, 1.0]])
    # warp It1 - find I(W(It1))
    rect = [0, 0, It.shape[0]-1, It.shape[1]-1]
    tempInt1 = RectBivariateSpline(np.linspace(0, It.shape[0], num=It.shape[0]), np.linspace(0, It.shape[1], num=It.shape[1]), It)
    It1Int = RectBivariateSpline(np.linspace(0, It1.shape[0], num=It1.shape[0]), np.linspace(0, It1.shape[1],num=It.shape[1]), It1)

    xi, yi = np.mgrid[rect[0]:rect[2]+1, rect[1]:rect[3]+1]
    tempInt = tempInt1.ev(yi, xi).T
    gradientX = tempInt1.ev(yi, xi, dx = 0, dy = 1).T
    gradientY = tempInt1.ev(yi, xi, dx = 1, dy = 0).T 

    grad = np.vstack((gradientX.flatten(), gradientY.flatten())).T
        
    # xi = xi.flatten()
    # yi = yi.flatten()
    gradientX = gradientX.flatten()
    gradientY = gradientY.flatten()

    steepestDesc = np.vstack((gradientX*xi.flatten(), gradientX*yi.flatten(), gradientX, gradientY*xi.flatten(), gradientY*yi.flatten(), gradientY)).T
    hessian = (steepestDesc.T @ steepestDesc)
    
    while True: 

        xi, yi = np.mgrid[rect[0]:rect[2]+1, rect[1]:rect[3]+1]
        x_warp = M[0][0] * xi + M[0][1] * yi + M[0][2]
        y_warp = M[1][0] * xi + M[1][1] * yi + M[1][2]


        warped1 = It1Int.ev(y_warp, x_warp).T 
        warped1 = warped1[0:It.shape[1],0:It.shape[0]]
        yWarpShape, xWarpShape = warped1.shape
        commonPts = (xWarpShape>0 and xWarpShape<It.shape[1] and yWarpShape>0 and yWarpShape<It.shape[0])
       
        # error image T - warped
        error = (warped1 - tempInt)*commonPts
        sDxError = np.dot(np.transpose(steepestDesc), error.flatten())
        delta_p = np.linalg.inv(hessian) @ sDxError
        p += delta_p
        delta_M = np.array([[delta_p[0]+1, delta_p[1], delta_p[2]], [delta_p[3], delta_p[4]+1, delta_p[5]],[0, 0, 1]])
        M = M @ np.linalg.inv(delta_M)
        i +=1

        if (np.linalg.norm(delta_p) <= threshold or i>num_iters):
            break
    # M = M[0:2,:]
    return M
