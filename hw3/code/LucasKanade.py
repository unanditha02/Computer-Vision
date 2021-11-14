import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import gradient
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    p = p0
    i = 0
    while True: 
        # warp It1 - find I(W(It1))
        tempInt = RectBivariateSpline(np.linspace(0, It.shape[0]-1, num=It.shape[0]), np.linspace(0, It.shape[1]-1, num=It.shape[1]), It)
        It1Int = RectBivariateSpline(np.linspace(0, It1.shape[0]-1, num=It1.shape[0]), np.linspace(0, It1.shape[1]-1,num=It.shape[1]), It1)
        # xi, yi = np.meshgrid(np.linspace(rect[0], rect[2], (rect[2]-rect[0]+1)), np.linspace(rect[1], rect[3], (rect[3]-rect[1]+1)), indexing='xy')
        xi, yi = np.mgrid[rect[0]:rect[2]+1, rect[1]:rect[3]+1]
        tempInt = tempInt.ev(yi, xi).T
        warped1 = It1Int.ev(yi+p[1], xi+p[0]).T 
 
        # error image T - warped
        error = tempInt - warped1
        # gradient 
        gradientX = It1Int.ev(yi+p[1], xi+p[0], dx = 0, dy = 1).T
        gradientY = It1Int.ev(yi+p[1], xi+p[0], dx = 1, dy = 0).T 

        grad = np.vstack((gradientX.flatten(), gradientY.flatten())).T
        # jacobian 
        jacobian = np.eye(2, dtype=int)
        steepestDesc = np.dot(grad, jacobian)
        
        hessian = np.dot(steepestDesc.T,  steepestDesc)
        sDxError = np.dot(np.transpose(steepestDesc), error.flatten())
        delta_p = np.linalg.inv(hessian) @ sDxError
        p += delta_p
        #print(delta_p, p)
        i +=1
        if (np.linalg.norm(delta_p) <= threshold or i>num_iters):
            break

    return p
