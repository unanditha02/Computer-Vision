import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from scipy.ndimage import affine_transform, binary_dilation, binary_erosion
import matplotlib.pyplot as plt
from InverseCompositionAffine import InverseCompositionAffine
def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    # M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    
    image2_warped = affine_transform(image2, np.linalg.inv(M))
    subImg = (image1 - image2_warped)

    for i in range(subImg.shape[0]):
        for j in range(subImg.shape[1]):
            mask[i,j] = 1 if subImg[i,j] > tolerance else 0

    kernelErode = np.ones((3,3), np.uint8)
    kernelDilate = np.ones((3,3), np.uint8)
    mask = binary_dilation(mask, kernelDilate)
    mask = binary_erosion(mask, kernelErode)
    
    
    return mask
