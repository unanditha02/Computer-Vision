import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
# plt.imshow(seq[:,:,0])
# plt.show()
rect = [59, 116, 145, 151]
It = seq[:, :, 0]
rectArray = np.array(rect)

x, y, _ = seq.shape
It1_rect = np.zeros(shape=(x,y,3))

for i in range(seq.shape[2]-1): #seq.shape[2]-
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]
    p = LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2))

    rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
    rectArray = np.vstack((rectArray, rect))
    
    for j in range(3):
        It1_rect[:,:,j] = It1
    It1_rect = cv2.rectangle(It1_rect, (int(rect[0]),int(rect[1])), (int(rect[2]),int(rect[3])), (0,0,255), 2)
    cv2.waitKey(1)
    cv2.imshow("Car Track", It1_rect)

    if(i == 0 or i == 100 or i == 200 or i == 300 or i == 400):
        plt.imshow(It1, cmap='gray')
        plt.axis('off')
        plt.gca().add_patch(patches.Rectangle((int(rect[0]),int(rect[3])), int(rect[2]-rect[0]), int(rect[1]-rect[3]), facecolor='none', edgecolor='red'))
        plt.show()
    print(i, rect)

np.save('carseqrects.npy', rectArray)