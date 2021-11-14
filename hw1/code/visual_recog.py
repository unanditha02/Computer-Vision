import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import visual_words
from sklearn.metrics import confusion_matrix

def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    histo, _ = np.histogram(wordmap.flatten(), bins = K, density = True, range=(0,K))

    return histo


    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    
    # ----- TODO -----
    img = Image.open(img_path)
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
        
    return feature

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
    
    K = opts.K
    L = opts.L

    # ----- TODO -----

    h, w = wordmap.shape
    h_rem = h % (2**L)
    w_rem = w % (2**L)
    if (h_rem != 0):
        wordmap = wordmap[0:h-h_rem]
        h = h-h_rem
    if (w_rem != 0):
        wordmap = wordmap[0:w-w_rem]
        w = w-w_rem

    hist_all = np.array([])
    weight = 2 ** -L

    for l in range(0,L+1):
        if l >=2:
            weight = 2 ** (l- L - 1)
            
        for i in range(0, h, int(h/(2**l))):
            for j in range(0, w, int(w/(2**l))):
                cell = wordmap[i:i+int(h/(2**l)),j:j+int(w/(2**l))]
                histo, _ = np.histogram(cell.flatten(), bins = K, density = True, range=(0,K))
                hist_all = np.append(hist_all, weight * histo/(4 ** l))

    return hist_all


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''
    # ----- TODO -----
    rows, columns = histograms.shape
    sum_min = np.zeros((rows, 1))
    for i in range(rows):
        for j in range(columns):
            sum_min[i] += min(word_hist[j], histograms[i][j])
    
    hist_dist = (np.ones(shape = sum_min.shape) - sum_min).flatten()

    return hist_dist

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    K = opts.K
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    feature_size = int(K * ((4 ** (SPM_layer_num+1)) - 1) / 3)
    features = np.zeros([len(train_files), feature_size])
    i=0
 
    for img_name in train_files:
        print("Train: ",  i, " ", img_name)
        img_path = join(data_dir, img_name)
        feature = get_image_feature(opts, img_path, dictionary)
        features[i,:] = feature
        i+=1

    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )


def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    train_labels = trained_system['labels']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----  

    test_pred = np.zeros((test_labels.shape))
    i=0
    for img_name in test_files:
        print("Test: ",  i, " ", img_name)
        img_path = join(data_dir, img_name)
        feature = get_image_feature(opts, img_path, dictionary)
        dist_hist = distance_to_set(feature, trained_system['features'])
        test_pred[i] = train_labels[np.argmin(dist_hist)]
        i+=1

    conf = confusion_matrix(test_labels, test_pred)
   
    accuracy = np.trace(conf) / np.sum(conf)
    return conf, accuracy
    

