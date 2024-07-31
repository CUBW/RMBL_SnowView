import os
import glob
import numpy as np
import time
import pickle

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
# from skimage.filters import roberts, sobel, scharr, prewitt
# from skimage.segmentation import felzenszwalb, quickshift, slic, watershed
# from skimage import filters
# from skimage.morphology import disk
from scipy import ndimage as nd
import cv2

SOURCE = "data/512_splits_4_channel/"
OUTPUT_DIR = "3_channel_results/"
TRAIN_BATCHES = 20

# def feature_extraction(img, mask, print_gabor = False):
#     if img.shape[0] !=4:
#         raise TypeError("Image must have 4 channels")
    
#     img = np.transpose(img, (1,2,0))
#     mask = np.transpose(mask, (1,2,0))

#     #generate grayscale
#     gray = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2GRAY)

#     #create output nparray
#     output = img.reshape((-1,4))
#     output.astype(np.uint8)
#     output = np.append(output, gray.reshape((-1,1)), axis=1)

#     # trim img to 3 channels
#     img = img[:,:,:3]

#     #gabor filter
#     # num = 1
#     # gabors = [5, 11, 23, 8, 6, 4]
#     # kernels = []
#     # for theta in range(2):
#     #     theta = theta / 4. * np.pi
#     #     for sigma in (1,3):
#     #         for lamda in np.arange(0, np.pi, np.pi/4):
#     #             for gamma in (.05, .5):
#     #                 if num in gabors:
#     #                     gabor_label = 'Gabor' + str(num)
#     #                     ksize = 9
#     #                     kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
#     #                     kernels.append(kernel)

#     #                     fimg = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
#     #                     filtered_img = fimg.reshape(-1)
#     #                     output = np.append(output, filtered_img.reshape((-1,1)), axis=1)
#     #                     if print_gabor:
#     #                         print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
#     #                 num += 1

#     #Roberts Edge
#     roberts_edge = roberts(gray).reshape(-1)
#     output = np.append(output, roberts_edge.reshape((-1,1)), axis=1)

#     #Sobel Edge
#     sobel_edge = sobel(gray).reshape(-1)
#     output = np.append(output, sobel_edge.reshape((-1,1)), axis=1)

#     #Scharr Edge
#     scharr_edge = scharr(gray).reshape(-1)
#     output = np.append(output, scharr_edge.reshape((-1,1)), axis=1)

#     #Prewitt Edge
#     prewitt_edge = prewitt(gray).reshape(-1)
#     output = np.append(output, prewitt_edge.reshape((-1,1)), axis=1)

#     # blurs
#     gaussian_img = nd.gaussian_filter(gray, sigma=3).reshape(-1)
#     output = np.append(output, gaussian_img.reshape((-1,1)), axis=1)

#     gaussian_img2 = nd.gaussian_filter(gray, sigma=7).reshape(-1)
#     output = np.append(output, gaussian_img2.reshape((-1,1)), axis=1)

#     median_img =  nd.median_filter(gray, size=3).reshape(-1)
#     output = np.append(output, median_img.reshape((-1,1)), axis=1)

#     # segmentation
#     # felzenszwalb
#     # segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
#     # output = np.append(output, segments_fz.reshape((-1,1)), axis=1)

#     # # quickshift
#     # segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
#     # output = np.append(output, segments_quick.reshape((-1,1)), axis=1)

#     # # slic
#     # segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
#     # output = np.append(output, segments_slic.reshape((-1,1)), axis=1)

#     # # watershed
#     # gradient = filters.rank.gradient(gray, disk(2))
#     # segments_ws = watershed(gradient, markers=250, compactness=0.001)
#     # output = np.append(output, segments_ws.reshape((-1,1)), axis=1)

#     # add labels last
#     output = np.append(output, mask.reshape((-1,1)), axis=1)

#     # using fourth column, delete everything that is 0
#     output = output[output[:,4] != 0]

#     #delete fourth column
#     output = np.delete(output, 4, 1)

#     return output

def load_data(img_filenames):
    for filename in img_filenames:
        # break name up into tags
        mask_name = filename.replace("images", "masks")
        
        # open files
        img_data = np.load(filename)
        mask_data = np.load(mask_name)
        
        # loop through arrays and process
        img_arrays = img_data.files
        mask_arrays = img_data.files
        
        random.shuffle(img_arrays)
        
        output = np.empty((0,4), dtype=np.uint8)
        # process data in chunks to limit number of array copying
        chunk = np.empty((0,4), dtype=np.uint8)
        chunk_size = 20 # number is a total guess but seems to work, each batch is ~ 3900 tiles
        
        for i in tqdm(range(len(img_arrays)), unit='tile'):
            # flatten images and mask
            flat_img = img_data[img_arrays[i]].reshape((-1,4))
            mask_img = mask_data[mask_arrays[i]].reshape((-1,1))
            
            if img_data[img_arrays[i]].shape != (4,512,512) or mask_data[mask_arrays[i]].shape != (1,512,512):
                continue
            
            # append together
            data = np.append(flat_img, mask_img, axis=1)
            
            # remove anything that is zero in fourth channel
            mask = data[:,3]!=0
            data = data[mask]
            
            # remove opacity column
            data = np.delete(data, 3, axis=1)
            
            # append to chunk
            chunk = np.vstack((chunk, data))
            
            if i%chunk_size >= chunk_size-1:
                output = np.append(output, chunk, axis=0)
                # reset chunk
                chunk = np.empty((0,4), dtype=np.uint8)
            
            if output.shape[0]>1000000:
                return output

        img_data.close()
        mask_data.close()
        
def plot_training_curve(img_filepaths):
    pass


def main():
    #create output dierctory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    start_time = time.time()
    
    # generate filenames
    img_filepaths = glob.glob(os.path.join(SOURCE, 'train*images.npz'), recursive=True)
    val_filepaths = glob.glob(os.path.join(SOURCE, 'val*images.npz'), recursive=True)
    test_filepaths = glob.glob(os.path.join(SOURCE, 'test*images.npz'), recursive=True)
    
    # train model
    rf = RandomForestClassifier(n_estimators=50)
    svm = SVC(kernel="rbf")
    linearsvc= LinearSVC()
    
    data = load_data(img_filenames=img_filepaths)
    
    # generate training curve
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
    
    common_params = {
    "X": data[:,:-1],
    "y": data[:,-1],
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": 4,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Accuracy",
    }
    
    for ax_idx, estimator in enumerate([rf,svm,linearsvc]):
        LearningCurveDisplay.from_estimator(estimator,  **common_params, ax=ax[ax_idx], verbose=2)
        handles, label = ax[ax_idx].get_legend_handles_labels()
        ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
        ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")
        
    plt.show()
    

if __name__ =="__main__":
    main()