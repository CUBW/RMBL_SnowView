import matplotlib.pyplot as plt
import cv2
import rasterio
import numpy as np
import os
import pandas as pd
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy import ndimage as nd
from skimage import filters
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.morphology import disk
from skimage.segmentation import felzenszwalb, quickshift, slic, watershed

DATA_DIRECTORY = "../data/640"
MODELS_DIRECTORY = "results/best_1/"
TEST_INDICES = [54, 12, 74, 123, 128, 105, 64, 68, 22, 36, 82, 104, 129, 53]
COLUMNS = [
    'Blue', 'Green', 'Red', 'Gray', 'Gabor4', 'Gabor5', 'Gabor6', 'Gabor8',
    'Gabor11', 'Gabor23', 'Roberts', 'Sobel', 'Scharr', 'Prewitt',
    'Gaussian s3', 'Gaussian s7', 'Median s3', 'Felzenszwalb', 'Quickshift',
    'SLIC', 'Watershed', 'labels'
]

def image_to_df(snow_path, mask_path, print_gabor=False, labeled=True):
    #load in images
    img = cv2.imread(snow_path)
    if labeled:
        mask = rasterio.open(mask_path)
    
    #generate grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #flatten image
    img2 = img.reshape((-1,3))

    #create dataframe
    df = pd.DataFrame(img2, columns=['Blue', 'Green', 'Red'])
    df['Gray'] = gray.reshape(-1)

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
                        df[gabor_label] = filtered_img
                        if print_gabor:
                            print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1

    #Roberts Edge
    roberts_edge = roberts(gray).reshape(-1)
    df['Roberts'] = roberts_edge

    #Sobel Edge
    sobel_edge = sobel(gray).reshape(-1)
    df['Sobel'] = sobel_edge

    #Scharr Edge
    scharr_edge = scharr(gray).reshape(-1)
    df['Scharr'] = scharr_edge

    #Prewitt Edge
    prewitt_edge = prewitt(gray).reshape(-1)
    df['Prewitt'] = prewitt_edge

    gaussian_img = nd.gaussian_filter(gray, sigma=3).reshape(-1)
    df['Gaussian s3'] = gaussian_img

    gaussian_img2 = nd.gaussian_filter(gray, sigma=7).reshape(-1)
    df['Gaussian s7'] = gaussian_img2

    median_img =  nd.median_filter(gray, size=3).reshape(-1)
    df['Median s3'] = median_img

    #segmentation
    #felzenszwalb
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    df['Felzenszwalb'] = segments_fz.reshape(-1)

    #quickshift
    segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    df['Quickshift'] = segments_quick.reshape(-1)

    #slic
    segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
    df['SLIC'] = segments_slic.reshape(-1)

    #watershed
    gradient = filters.rank.gradient(gray, disk(2))
    segments_ws = watershed(gradient, markers=250, compactness=0.001)
    df['Watershed'] = segments_ws.reshape(-1)

    #labels
    if labeled:
        df['labels'] = mask.read(1).reshape(-1)

    return df

def load_images(csv_df):
    """
    Loads and preprocess images from paths given in a dataframe.

    Args:
        csv_df (pd.DataFrame): The CSV dataframe containing the paths to the images and masks.

    Returns:
        tuple: A tuple containing the list of dataframes and a list of tuples of image and mask paths.

    """
    master_df = []
    image_paths = []
    # loop through all test indices
    for i in TEST_INDICES:
        # get the image path and mask path
        img_path = os.path.join("../", csv_df['snow_path'][i])
        mask_path = os.path.join("../", csv_df['snowbinary_path'][i])

        # load them through preprocessing
        image_df = image_to_df(img_path, mask_path)

        # append to master dataframe
        master_df.append(image_df)
        image_paths.append((img_path, mask_path))

    return master_df, image_paths

def analyze_best(model, image_df, image_paths):
    """
    Analyzes the performance of a given model on a test dataset.

    Args:
        model (object): The trained model to be evaluated.
        test_df (list): The test dataset containing features and labels.
        image_paths (list): List of image paths and corresponding mask paths.

    Returns:
        tuple: A tuple containing two lists - accuracies and confusion matrices.
            - accuracies (list): List of accuracy scores for each image.
            - confusion_matrices (list): List of confusion matrices for each image.
    """

    # maeke lists to store accuracies and confusion matrices
    accuracies = []
    confusion_matrices = []
    precisions = []
    recalls = []
    f1s = []

    for curr_df in test_df: 
        # split curr_df into data and labels
        curr_x = curr_df.drop(columns=['labels'])
        curr_y = curr_df['labels']
        
        # get prediction from model
        y_pred = model.predict(curr_x)

        # get accuracy
        accuracy = accuracy_score(curr_y, y_pred)
        accuracies.append(accuracy)

        precisions.append(precision_score(curr_y, y_pred))
        recalls.append(recall_score(curr_y, y_pred))
        f1s.append(f1_score(curr_y, y_pred))

        # get confusion matrix
        cm = confusion_matrix(curr_y, y_pred)
        confusion_matrices.append(cm)
    
    # print average accuracy
    print(f"Average Accuracy: {sum(accuracies)/len(accuracies)}")
    print(f"Average Precision: {sum(precisions)/len(precisions)}")
    print(f"Average Recall: {sum(recalls)/len(recalls)}")
    print(f"Average F1: {sum(f1s)/len(f1s)}")
    

    return accuracies, confusion_matrices


if __name__ =="__main__":
    # open csv
    csv_df = pd.read_csv(os.path.join(DATA_DIRECTORY, "640.csv"))

    # load images
    test_df, image_paths = load_images(csv_df) 

    best_model_index = 5

    # load model
    model = pickle.load(open(os.path.join(MODELS_DIRECTORY, f"model_{best_model_index}.pkl"), "rb"))

    # analyze model
    accuracies, confusion_matrices = analyze_best(model, test_df, image_paths)