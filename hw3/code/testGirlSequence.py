import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import cv2
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

It = seq[:, :, 0]
plt.imshow(It)
plt.gca().add_patch(patches.Rectangle((rect[0],rect[3]), rect[2]-rect[0], rect[1]-rect[3],facecolor='none',
                    edgecolor='red'))
plt.show()
p0 = np.zeros(2)
rectArray = np.array(rect)
x, y, _ = seq.shape
# It1_rect = np.zeros(shape=(x,y,3))
for i in range(seq.shape[2]-1):
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]
    p = LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2))
    
    rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
    rectArray = np.vstack((rectArray, rect))

    # for j in range(3):
    #     It1_rect[:,:,j] = It1
    # It1_rect = cv2.rectangle(It1_rect, (int(rect[0]),int(rect[1])), (int(rect[2]),int(rect[3])), (0,0,255), 2)
    # cv2.waitKey(1)
    # cv2.imshow("Girl Track", It1_rect)
    
    if(i == 0 or i == 20 or i == 40 or i == 60 or i == 80):
        plt.imshow(It1, cmap = 'gray')
        plt.axis('off')
        plt.gca().add_patch(patches.Rectangle((int(rect[0]),int(rect[3])), int(rect[2]-rect[0]), int(rect[1]-rect[3]), facecolor='none', edgecolor='red'))
        plt.show()
        
    print(i, rect)
np.save('girlseqrects.npy', rectArray)