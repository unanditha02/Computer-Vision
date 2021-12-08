# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
import matplotlib.pyplot as plt

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    U, Sigma, Vh = np.linalg.svd(I, full_matrices=False)
    Sigma[3:] = 0
    # S = np.diag(np.sqrt(Sigma))
    S = np.diag(Sigma)
    
    L_full = (U @ S)
    L_trans = L_full[:,0:3]
    L = L_trans.T

    # B_full = (S @ Vh)
    # B = B_full[0:3,:]
    # I_cap = (U @ S) @ Vh
    # B = np.linalg.inv(L @ L.T) @ L @ I_cap
    B = Vh[0:3,:]
    return B, L


if __name__ == "__main__":

    # Put your main code here
    
    I, L0, s = loadData()
    print('L0: ', L0)

    B, L = estimatePseudonormalsUncalibrated(I)
    print('L: ', L)

    albedos, normals = estimateAlbedosNormals(B)
    normals = enforceIntegrability(normals, s, sig = 3)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    plt.imshow(albedoIm, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.imshow(normalIm, cmap='rainbow')
    plt.axis('off')
    plt.show()

    surface = estimateShape(normals, s)
    plotSurface(surface)

    # q 2e
    mu = 1
    v = 1
    lamda = 1
    G = np.array([[1, 0, 0], [0, 1, 0], [mu, v, lamda]])

    normalsG = np.linalg.inv(G.T) @ normals
    albedoImG, normalImG = displayAlbedosNormals(albedos, normalsG, s)

    plt.imshow(albedoImG, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.imshow(normalImG, cmap='rainbow')
    plt.axis('off')
    plt.show()

    surfaceG = estimateShape(normalsG, s)
    plotSurface(surfaceG)


