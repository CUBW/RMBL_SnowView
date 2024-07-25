import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.segmentation import felzenszwalb, quickshift, slic, watershed
from skimage import filters
from skimage.morphology import disk
from scipy import ndimage as nd

SOURCE_DIR = "data/512_splits"
OUTPUT_DIR = "results"

def feature_extraction(img, mask, print_gabor = False):
    img = np.transpose(img, (1,2,0))
    mask = np.transpose(mask, (1,2,0))

    #generate grayscale
    gray = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2GRAY)

    #create output nparray
    output = img.reshape((-1,3))
    output.astype(np.uint8)
    output = np.append(output, gray.reshape((-1,1)), axis=1)

    #gabor filter
    num = 1
    gabors = [5, 11, 23, 8, 6, 4]
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1,3):
            for lamda in np.arange(0, np.pi, np.pi/4):
                for gamma in (.05, .5):
                    if num in gabors:
                        gabor_label = 'Gabor' + str(num)
                        ksize = 9
                        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                        kernels.append(kernel)

                        fimg = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                        filtered_img = fimg.reshape(-1)
                        output = np.append(output, filtered_img.reshape((-1,1)), axis=1)
                        if print_gabor:
                            print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1

    #Roberts Edge
    roberts_edge = roberts(gray).reshape(-1)
    output = np.append(output, roberts_edge.reshape((-1,1)), axis=1)

    #Sobel Edge
    sobel_edge = sobel(gray).reshape(-1)
    output = np.append(output, sobel_edge.reshape((-1,1)), axis=1)

    #Scharr Edge
    scharr_edge = scharr(gray).reshape(-1)
    output = np.append(output, scharr_edge.reshape((-1,1)), axis=1)

    #Prewitt Edge
    prewitt_edge = prewitt(gray).reshape(-1)
    output = np.append(output, prewitt_edge.reshape((-1,1)), axis=1)

    # blurs
    gaussian_img = nd.gaussian_filter(gray, sigma=3).reshape(-1)
    output = np.append(output, gaussian_img.reshape((-1,1)), axis=1)

    gaussian_img2 = nd.gaussian_filter(gray, sigma=7).reshape(-1)
    output = np.append(output, gaussian_img2.reshape((-1,1)), axis=1)

    median_img =  nd.median_filter(gray, size=3).reshape(-1)
    output = np.append(output, median_img.reshape((-1,1)), axis=1)

    # segmentation
    # felzenszwalb
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    output = np.append(output, segments_fz.reshape((-1,1)), axis=1)

    # quickshift
    segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    output = np.append(output, segments_quick.reshape((-1,1)), axis=1)

    # slic
    segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
    output = np.append(output, segments_slic.reshape((-1,1)), axis=1)

    # watershed
    gradient = filters.rank.gradient(gray, disk(2))
    segments_ws = watershed(gradient, markers=250, compactness=0.001)
    output = np.append(output, segments_ws.reshape((-1,1)), axis=1)

    # add labels last
    output = np.append(output, mask.reshape((-1,1)), axis=1)

    # using fourth column, delete everything that is 0
    output = output[output[:,4] != 0]

    #delete fourth column
    output = np.delete(output, 4, 1)

    return output


if __name__ == "__main__":
    # create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # open npz file
    testimg = np.load('output_4_channel/DeerCreekTrail_2019_05_22_tiles_images.npz')
    testmask = np.load('output_4_channel/DeerCreekTrail_2019_05_22_tiles_masks.npz')

    img_arrays = testimg.files
    mask_arrays = testmask.files

    # run feature extraction on 4
    features = feature_extraction(testimg[img_arrays[4]], testmask[mask_arrays[4]])
