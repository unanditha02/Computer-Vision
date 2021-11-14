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
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
carseqrects = np.load("carseqrects.npy")
# 1st template
rect0 = rect
It0 = seq[:, :, 0]
rectArray = np.array(rect)

x, y, _ = seq.shape
It1_rect = np.zeros(shape=(x,y,3))

for i in range(seq.shape[2]-1):
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]
    p = LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2))
    rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
    
    pn = p + [rect[0]-rect0[0], rect[1]-rect0[1]]
    p_star = LucasKanade(It0, It1, rect0, threshold, num_iters, p0=pn)
    
    if(np.linalg.norm(p_star-pn) <= template_threshold):
        p_star = p_star-[rect[0]-rect0[0], rect[1]-rect0[1]]
        rect = [rect[0]+p_star[0], rect[1]+p_star[1], rect[2]+p_star[0], rect[3]+p_star[1]]
    else:
        rect=rect
    rectArray = np.vstack((rectArray, rect))
    rectn = carseqrects[i+1]

    # for j in range(3):
    #     It1_rect[:,:,j] = It1
    # It1_rect = cv2.rectangle(It1_rect, (int(rect[0]),int(rect[1])), (int(rect[2]),int(rect[3])), (0,0,255))
    # It1_rect = cv2.rectangle(It1_rect, (int(rectn[0]),int(rectn[1])), (int(rectn[2]),int(rectn[3])), (255,0,0))
    # cv2.waitKey(1)
    # cv2.imshow("Car Track", It1_rect)

    # if(i == 0 or i == 100 or i == 200 or i == 300 or i == 400):
    #     plt.imshow(It1, cmap='gray')
    #     plt.gca().add_patch(patches.Rectangle((int(rect[0]),int(rect[3])), int(rect[2]-rect[0]), int(rect[1]-rect[3]), facecolor='none', edgecolor='red'))
    #     plt.gca().add_patch(patches.Rectangle((int(rectn[0]),int(rectn[3])), int(rectn[2]-rectn[0]), int(rectn[1]-rectn[3]), facecolor='none', edgecolor='blue'))
    #     plt.axis('off')
    #     plt.show()
    print(i, rect)

np.save('carseqrects-wcrt.npy', rectArray)