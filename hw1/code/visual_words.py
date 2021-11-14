import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
from skimage.util.dtype import img_as_float64
import math
from random import seed, randint
import util
from sklearn.cluster import KMeans

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    filter_scales = opts.filter_scales
    # ----- TODO -----
    img_lab = np.ndarray(np.shape(img))
    if(len(np.shape(img)) < 3):
        img = np.dstack((img, img, img))

    img_lab = skimage.color.rgb2lab(img)
    img1, img2, img3 = img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]
   
    gauss1 = np.dstack((scipy.ndimage.gaussian_filter(img1, sigma=1), scipy.ndimage.gaussian_filter(img2, sigma=1), scipy.ndimage.gaussian_filter(img3, sigma=1))) 
    gauss2 = np.dstack((scipy.ndimage.gaussian_filter(img1, sigma=2), scipy.ndimage.gaussian_filter(img2, sigma=2), scipy.ndimage.gaussian_filter(img3, sigma=2))) 
    gauss3 = np.dstack((scipy.ndimage.gaussian_filter(img1, sigma=4), scipy.ndimage.gaussian_filter(img2, sigma=4), scipy.ndimage.gaussian_filter(img3, sigma=4))) 
    gauss4 = np.dstack((scipy.ndimage.gaussian_filter(img1, sigma=8), scipy.ndimage.gaussian_filter(img2, sigma=8), scipy.ndimage.gaussian_filter(img3, sigma=8))) 
    gauss5 = np.dstack((scipy.ndimage.gaussian_filter(img1, sigma=(8*math.sqrt(2))), scipy.ndimage.gaussian_filter(img2, sigma=(8*math.sqrt(2))), scipy.ndimage.gaussian_filter(img3, sigma=(8*math.sqrt(2))))) 
    log1 = np.dstack((scipy.ndimage.gaussian_laplace(img1, sigma=1), scipy.ndimage.gaussian_laplace(img2, sigma=1), scipy.ndimage.gaussian_laplace(img3, sigma=1)))
    log2 = np.dstack((scipy.ndimage.gaussian_laplace(img1, sigma=2), scipy.ndimage.gaussian_laplace(img2, sigma=2), scipy.ndimage.gaussian_laplace(img3, sigma=2)))
    log3 = np.dstack((scipy.ndimage.gaussian_laplace(img1, sigma=4), scipy.ndimage.gaussian_laplace(img2, sigma=4), scipy.ndimage.gaussian_laplace(img3, sigma=4)))
    log4 = np.dstack((scipy.ndimage.gaussian_laplace(img1, sigma=8), scipy.ndimage.gaussian_laplace(img2, sigma=8), scipy.ndimage.gaussian_laplace(img3, sigma=8)))
    log5 = np.dstack((scipy.ndimage.gaussian_laplace(img1, sigma=(8*math.sqrt(2))), scipy.ndimage.gaussian_laplace(img2, sigma=(8*math.sqrt(2))), scipy.ndimage.gaussian_laplace(img3, sigma=(8*math.sqrt(2)))))
    dogx1 = np.dstack((scipy.ndimage.gaussian_filter(img1, sigma=1, order = (0, 1)), scipy.ndimage.gaussian_filter(img2, sigma=1, order = (0, 1)), scipy.ndimage.gaussian_filter(img3, sigma=1, order = (0, 1)))) 
    dogx2 = np.dstack((scipy.ndimage.gaussian_filter(img1, sigma=2, order = (0, 1)), scipy.ndimage.gaussian_filter(img2, sigma=2, order = (0, 1)), scipy.ndimage.gaussian_filter(img3, sigma=2, order = (0, 1)))) 
    dogx3 = np.dstack((scipy.ndimage.gaussian_filter(img1, sigma=4, order = (0, 1)), scipy.ndimage.gaussian_filter(img2, sigma=4, order = (0, 1)), scipy.ndimage.gaussian_filter(img3, sigma=4, order = (0, 1)))) 
    dogx4 = np.dstack((scipy.ndimage.gaussian_filter(img1, sigma=8, order = (0, 1)), scipy.ndimage.gaussian_filter(img2, sigma=8, order = (0, 1)), scipy.ndimage.gaussian_filter(img3, sigma=8, order = (0, 1)))) 
    dogx5 = np.dstack((scipy.ndimage.gaussian_filter(img1, sigma=(8*math.sqrt(2)), order = (0, 1)), scipy.ndimage.gaussian_filter(img2, sigma=(8*math.sqrt(2)), order = (0, 1)), scipy.ndimage.gaussian_filter(img3, sigma=(8*math.sqrt(2)), order = (0, 1)))) 
    dogy1 = np.dstack((scipy.ndimage.gaussian_filter(img1, sigma=1, order = (1, 0)), scipy.ndimage.gaussian_filter(img2, sigma=1, order = (1, 0)), scipy.ndimage.gaussian_filter(img3, sigma=1, order = (1, 0)))) 
    dogy2 = np.dstack((scipy.ndimage.gaussian_filter(img1, sigma=2, order = (1, 0)), scipy.ndimage.gaussian_filter(img2, sigma=2, order = (1, 0)), scipy.ndimage.gaussian_filter(img3, sigma=2, order = (1, 0)))) 
    dogy3 = np.dstack((scipy.ndimage.gaussian_filter(img1, sigma=4, order = (1, 0)), scipy.ndimage.gaussian_filter(img2, sigma=4, order = (1, 0)), scipy.ndimage.gaussian_filter(img3, sigma=4, order = (1, 0)))) 
    dogy4 = np.dstack((scipy.ndimage.gaussian_filter(img1, sigma=8, order = (1, 0)), scipy.ndimage.gaussian_filter(img2, sigma=8, order = (1, 0)), scipy.ndimage.gaussian_filter(img3, sigma=8, order = (1, 0)))) 
    dogy5 = np.dstack((scipy.ndimage.gaussian_filter(img1, sigma=(8*math.sqrt(2)), order = (1, 0)), scipy.ndimage.gaussian_filter(img2, sigma=(8*math.sqrt(2)), order = (1, 0)), scipy.ndimage.gaussian_filter(img3, sigma=(8*math.sqrt(2)), order = (1, 0)))) 

    filter_responses = np.dstack((gauss1, log1, dogx1, dogy1, gauss2, log2, dogx2, dogy2, gauss3, log3, dogx3, dogy3, gauss4, log4, dogx4, dogy4, gauss5, log5, dogx5, dogy5))

    return filter_responses

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    pass

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    alpha = opts.alpha
    # ----- TODO -----
    filter_responses_alphaT = np.zeros(shape = (alpha*len(train_files), 60))
    t = 0
    for img_name in train_files:
        img_path = join(opts.data_dir, img_name)
        img = Image.open(img_path)
        img = np.array(img).astype(np.float32)/255
        filter_responses = extract_filter_responses(opts, img)
  
        filter_responses_alpha = np.zeros(shape = (alpha, filter_responses.shape[2]))

        for i in range(alpha):
            #alpha x 3F
            filter_responses_alpha[i, :] = filter_responses[randint(0, filter_responses.shape[0]-1), randint(0, filter_responses.shape[1]-1), :].reshape(1, filter_responses.shape[2])

        filter_responses_alphaT[t:t+alpha] = filter_responses_alpha
        t += alpha
        print("Dictionary train: ", img_name)

    kmeans = KMeans(n_clusters=K).fit(filter_responses_alphaT)
    dictionary = kmeans.cluster_centers_

    ## example code snippet to save the dictionary
    print(dictionary.shape)
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    filter_responses = extract_filter_responses(opts, img)
    wordmap = np.zeros(shape = (np.shape(img)[0], np.shape(img)[1]))
    for i in range(filter_responses.shape[0]):
        for j in range(filter_responses.shape[1]):
            pixel_array = filter_responses[i, j, :].reshape(1, filter_responses.shape[2])
            wordmap[i, j] = np.argmin((scipy.spatial.distance.cdist(pixel_array, dictionary, metric='euclidean')), axis=1)

    return wordmap