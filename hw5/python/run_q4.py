import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import pickle
import string
from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    
    plt.imshow(bw, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    # # find the rows using..RANSAC, counting, clustering, etc.
    # ##########################
    # ##### your code here #####
    # ##########################
    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    heights = [bbox[2]-bbox[0] for bbox in bboxes]
    mean_height = sum(heights)/len(heights)
        # sort the bounding boxes with center y
    centers = [((bbox[2]+bbox[0])//2, (bbox[3]+bbox[1])//2, bbox[2]-bbox[0], bbox[3]-bbox[1]) for bbox in bboxes]
    centers = sorted(centers, key=lambda p: p[0])
    rows = []
    pre_h = centers[0][0]
        # cluster rows
    row = []
    for c in centers:
        if c[0] > pre_h + mean_height:
                row = sorted(row, key=lambda p: p[1])
                rows.append(row)
                row = [c]
                pre_h = c[0]
        else:
                row.append(c)
    row = sorted(row, key=lambda p: p[1])
    rows.append(row)

    data = []
    for row in rows:
        row_data = []
        bbox_row = np.array(row)
        for bbox in bbox_row:
            # crop out the character
            y, x, h, w = bbox
            crop = bw[y-h//2:y+h//2, x-w//2:x+w//2]
                # pad it to square
            h_pad, w_pad = 0, 0
            if h > w:
                    h_pad = h//20
                    w_pad = (h-w)//2+h_pad
            elif h < w:
                    w_pad = w//20
                    h_pad = (w-h)//2+w_pad
            crop = np.pad(crop, ((h_pad, h_pad), (w_pad, w_pad)), 'constant', constant_values=(1, 1))
            crop = skimage.transform.resize(crop, (32, 32), anti_aliasing=False)
            crop = np.transpose(crop)
            # plt.imshow(crop, cmap='gray')
            # plt.show()
            row_data.append(crop.flatten())
        data.append(np.array(row_data))
    
    # load the weights
    # run the crops through your neural network and print them out
    
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################
    pred = []

    for i in range(len(data)):
        h1 = forward(data[i], params, 'layer1')
        probs = forward(h1, params, 'output',softmax)
        for j in range(len(data[i])):
            pred.append(letters[np.argmax(probs[j,:])])
    print("Image ", img)
    print(pred)