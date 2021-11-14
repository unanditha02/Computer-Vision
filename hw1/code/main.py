from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts


def main():
    opts = get_opts()

    # Q1.1
    img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    filter_responses = visual_words.extract_filter_responses(opts, img)
    util.display_filter_responses(opts, filter_responses)
    print(opts.K, opts.L, opts.alpha)

    # Q1.2
    n_cpu = util.get_num_CPU()
    # visual_words.compute_dictionary(opts, n_worker=n_cpu)
    
    # Q1.3
    # img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # util.visualize_wordmap(wordmap)
    
    # #Q2.2
    # histo = visual_recog.get_feature_from_wordmap(opts, wordmap)
    # word_hist = visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)
    # hist_all = visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)
    # print(hist_all.shape)
    # print(np.sum(hist_all))
    # #Q2.3
    # data_dir = opts.data_dir
    # train_files = open(join(data_dir, 'train_files_small.txt')).read().splitlines()
    # histograms = np.zeros([len(train_files), 50])
    # i=0
    # # histo = np.array([])
    # for img_name in train_files:
    #     print(img_name)
    #     img_path = join(data_dir, img_name)
    #     img = Image.open(img_path)
    #     wordmap = visual_words.get_visual_words(opts, img, dictionary)
    #     hist = visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)
    #     # histo = np.append(histo, [hist], axis=0)
    #     histograms[i,:] = hist
    #     i+=1
    
    # print(histograms.shape, word_hist.shape)
    # print(histo.shape)
    # print(visual_recog.distance_to_set(word_hist, histograms))

    # Q2.1-2.4
    # n_cpu = util.get_num_CPU()
    # visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    # Q2.5
    # n_cpu =  util.get_num_CPU()
    # conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    
    # print(conf)
    # print(accuracy)
    # print("Accuracy: ", accuracy*100)
    # np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    # np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
