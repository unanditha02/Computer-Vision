import numpy as np
import cv2
import os
#Import necessary functions
import multiprocessing as mp
from loadVid import loadVid
from planarH import computeH_ransac
from planarH import compositeH
from matchPics import matchPics
from opts import get_opts
opts = get_opts()

# Function to process each frame and create composite frame
def compositeFrame(f, opts, ar_frame_width):
    print("Processing Frame... ", f)
    # Read source and destination frames
    frame_ar = ar_frames[f, :, :, :]
    frame_book = book_frames[f, :, :, :]
    
    # crop template image 
    center_x = (ar_frame_width / 2)
    crop_frame_ar = frame_ar[44:316, int(center_x - cv_cover_width/2):int(center_x + cv_cover_width/2)]
    resize_frame_ar = cv2.resize(crop_frame_ar,(cv_cover_width, cv_cover_height))
    # find matches between destination and template frame
    matches, locs1, locs2 = matchPics(cv_cover, frame_book, opts)

    x1 = np.zeros(shape=(len(matches), 2))
    x2 = np.zeros(shape=(len(matches), 2))
    for i in range(len(matches)):
        x1[i] = locs1[matches[i][0]]
        x2[i] = locs2[matches[i][1]]

    x1[:, [0, 1]] = x1[:, [1, 0]]
    x2[:, [0, 1]] = x2[:, [1, 0]]
    
    try:
        if len(x1) > 4:
            # Compute H
            bestH2to1, max_inliers = computeH_ransac(x2, x1, opts)
            composite_img = compositeH(bestH2to1, resize_frame_ar, frame_book)

        else:
            composite_img = frame_book
    except:
        print("Error in: ", f)
        print("Values: ", len(x1), ", ", len(matches))
        composite_img = frame_book

    # save processed frames 
    frame_write_path = frame_path+str(f)+".png"
    cv2.imwrite(frame_write_path, composite_img)

# File paths
book_path = '../data/book.mov'
ar_path = '../data/ar_source.mov'
cv_cover = cv2.imread('../data/cv_cover.jpg')
ar_result_path = '../data/ar.avi'
frame_path = '../frames/frame_'

#Write script for Q3.1
book_frames = loadVid(book_path)
ar_frames = loadVid(ar_path)
cv_cover_height, cv_cover_width, _ = cv_cover.shape
print("Load frames...")

frame0_ar = ar_frames[0, :, :, :]
frame0_book = book_frames[0, :, :, :]
ar_frame_height, ar_frame_width, _ = frame0_ar.shape
book_frame_height, book_frame_width, _ = frame0_book.shape
print("Frame size... ", ar_frame_height, ar_frame_width, book_frame_height, book_frame_width)
pool = mp.Pool(mp.cpu_count())


print("Starting starmap...")
# multiprocess to process video frames
input_args = [(f, opts, ar_frame_width) for f in range(len(ar_frames))] 
result = pool.starmap(compositeFrame, input_args)

# to write video using saved frames
_, _, files = next(os.walk("../frames"))
file_count = len(files)
frame0 = cv2.imread(frame_path + str(0) + ".png")

h, w, _ = frame0.shape
out = cv2.VideoWriter(ar_result_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (w, h))

print("Number of frames saved... ", file_count)
print("Writing frames...")
for i in range(file_count):
    frame = cv2.imread(frame_path+str(i)+".png")
    out.write(frame)

out.release()