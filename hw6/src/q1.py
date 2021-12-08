# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from utils import integrateFrankot
from skimage.color import rgb2xyz
from skimage.io import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    nx, ny = res
    image = np.zeros((ny, nx))
    x = np.arange(int(-nx/2), int(nx/2), 1)
    y = np.arange(int(-ny/2), int(ny/2), 1)
    xx, yy = np.meshgrid(x,y)
    xx = xx*pxSize
    yy = yy*pxSize
    isit = (xx**2 + yy**2) <= rad**2

    zz = np.sqrt(rad**2 - xx**2 - yy**2)
    zz[~isit] = 0
    sphere  = np.dstack((xx, yy, zz))
    
    
    image = np.dot(sphere,light)
    
    image = np.where(image < 0, 0, image)
    image[~isit] = 0

    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    # Extracting image shape s
    image = plt.imread(path+'input_1.tif')
    image = np.array(image, dtype='uint16')
    s = (image.shape[0],image.shape[1])
    I = np.zeros((7,s[0]*s[1]))

    # Reading images, converting to XYZ format, extracting luminance and flattening
    for i in range(7):
        filepath = path+'input_'+str(i+1)+'.tif'
        image =  imread(filepath)
        img_xyz = rgb2xyz(image)
        I[i,:] = img_xyz[:,:,1].flatten()

    # Reading the lighting directions data 
    sources = np.load(path+'sources.npy')
    L = sources.T

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = np.linalg.inv(L @ L.T) @ L @ I
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = np.linalg.norm(B, axis=0)
    normals = np.divide(B, albedos)
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = albedos.reshape((s))
    x,y = s
    normalIm = normals.T.reshape((x,y,3))

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """
    zx = (-normals[0,:]/normals[2,:]).reshape((s))
    zy = (-normals[1,:]/normals[2,:]).reshape((s))
    
    surface = integrateFrankot(zx, zy)
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    x, y = surface.shape

    X, Y = np.meshgrid(np.arange(0, y, 1), np.arange(0, x, 1))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, -surface, cmap='coolwarm')
    plt.show()


if __name__ == '__main__':
    
    # Put your main code here

    center = [0, 0, 0]
    rad = 75e-4
    pxSize = 7e-6
    light = np.divide([[1, 1, 1],[1, -1, 1],[-1, -1, 1]], math.sqrt(3))
    res = (3840, 2160)
    for i in range(3):
        image = renderNDotLSphere(center, rad, light[i], pxSize, res)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

    
    I, L, s = loadData()
    print('Shapes for I, L, s: ', I.shape, L.shape, s)

    U, S, Vh = np.linalg.svd(I, full_matrices=False)
    print(np.linalg.matrix_rank(I))
    # print(u)
    print('Singular Values: ', S)
    # print(vh)

    B = estimatePseudonormalsCalibrated(I, L)
    print('B: ', B)

    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    plt.imshow(albedoIm, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.imshow(normalIm, cmap='rainbow')
    plt.axis('off')
    plt.show()

    surface = estimateShape(normals, s)
    plotSurface(surface)