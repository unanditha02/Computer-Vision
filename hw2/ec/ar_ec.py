import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/nanditha/hw2/python/')
import numpy as np
import cv2
import os
import time
#Import necessary functions
import multiprocessing as mp
from loadVid import loadVid
from planarH import computeH_ransac
from planarH import compositeH
from matchPics import matchPics
from opts import get_opts
opts = get_opts()

# Function to process each frame and create composite frame
def compositeFrame(frame_ar, frame_book, opts):
    _, ar_frame_width, _ = frame_ar.shape

    # crop template image 
    center_x = (ar_frame_width / 2)
    crop_frame_ar = frame_ar[44:316, int(center_x - cv_cover_width/2):int(center_x + cv_cover_width/2)]
    resize_frame_ar = cv2.resize(crop_frame_ar,(cv_cover_width, cv_cover_height))
    # ORB
    orb = cv2.ORB_create(nfeatures=200)
    kp1, des1 = orb.detectAndCompute(cv_cover, None)
    kp2, des2 = orb.detectAndCompute(frame_book, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    numGoodMatches = int(len(matches) * 0.30)
    matches = matches[:numGoodMatches]

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    try:
        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
        composite_img = compositeH(H, resize_frame_ar, frame_book)

    except:
        print("Error")
        print("Values: ", len(points1), ", ", len(matches))
        composite_img = frame_book
    
    return composite_img

# File paths
book_path = '../data/book.mov'
ar_path = '../data/ar_source.mov'
cv_cover = cv2.imread('../data/cv_cover.jpg')

cv_cover_height, cv_cover_width, _ = cv_cover.shape
print("Load frames...")

cap_ar = cv2.VideoCapture(ar_path)
cap_book = cv2.VideoCapture(book_path) 
i = 0
# to calculate fps
prev_frame_time = 0
new_frame_time = 0
ret_ar = True

while(ret_ar): 
    time.sleep(0.01)
    # Capture frame-by-frame 
    ret_ar, frame_ar = cap_ar.read() 
    _, frame_book = cap_book.read()
    composite_frame = compositeFrame(frame_ar, frame_book, opts)
  
    cv2.imshow('AR Video', composite_frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 
    i += 1
    # FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    print("FPS... ", fps)

cap_ar.release() 
cap_book.release()
cv2.destroyAllWindows() 