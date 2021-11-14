import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from LucasKanadeAffine import LucasKanadeAffine
from SubtractDominantMotion import SubtractDominantMotion
from InverseCompositionAffine import InverseCompositionAffine
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e2, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')

x_, y_, _ = seq.shape
It1_3 = np.zeros(shape=(x_,y_,3))

for i in range(10):#seq.shape[2]-1):
    It = seq[:, :, 0]
    It1 = seq[:, :, i+1]
    mask = SubtractDominantMotion(It, It1, threshold, num_iters, tolerance)

    # to add blue patches to indicate motion
    for j in range(3):
        It1_3[:,:,j] = It1
    for x in range(x_):
        for y in range(y_):
            
            if mask[x,y] == 1:
                It1_3[x,y,0] = 0
                It1_3[x,y,1] = 0
                It1_3[x,y,2] = 1
    plt.imshow(It1_3, cmap='gray')
    plt.axis('off')
    plt.show()
    print(i)
    # if(i == 30 or i == 60 or i == 90 or i == 120):
    #     plt.imshow(It1_3, cmap='gray')
    #     plt.axis('off')
    #     plt.show()