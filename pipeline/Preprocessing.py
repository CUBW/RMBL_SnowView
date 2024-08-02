import rasterio
import numpy as np
import pandas as pd
import cv2

from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.segmentation import felzenszwalb, quickshift, slic, watershed
from skimage import filters
from skimage.morphology import disk
from scipy import ndimage as nd

def nn_preprocessing(filepath, channels=4, datatype='uint8', **kwargs):
    """
    Preprocesses an image for neural network input.
    Height and Width default to 640 by 640.

    Args:
        filepath (str): The path to the image file.
        channels (int, optional): The number of channels in the image. Defaults to 4.
        datatype (str, optional): The desired datatype for the image. Can be 'uint8' or 'float32'. Defaults to 'uint8'.
        **kwargs: Additional keyword arguments for image resizing.

    Returns:
        tuple: A tuple containing the preprocessed image data and the updated image metadata.
    """
    
    # read kwargs
    if 'height' in kwargs:
        height = kwargs['height']
    else:
        height = 640
    if 'width' in kwargs:
        width = kwargs['width']
    else:
        width = 640

    # open the image
    img = rasterio.open(filepath)
    img_meta = img.meta

    # read the image
    img_data = img.read()

    # change any nan to 0
    img_data = np.nan_to_num(img_data, nan=0)

    # fix datatypes
    if datatype == 'uint8':
        img_data = img_data.astype(np.uint8)
    elif datatype == 'float32':
        img_data = (img_data/255.0).astype(np.float32)
    else:
        raise ValueError('Datatype not recognized')

    # check number of channels
    if img_data.shape[0] != channels:
        # add in a fourth channel
        img_data = np.append(img_data, np.zeros((1, img_data.shape[1], img_data.shape[2])), axis=0)

        # set all data for fourth channel
        img_data[3] = (img_data[0] != 0) | (img_data[1] != 0) | (img_data[2] != 0)

    # change size of image
    if height is None:
        # preserve aspect ratio
        height = int(width * img_data.shape[1] / img_data.shape[2])

    img_data = cv2.resize(img_data, (width, height), interpolation=cv2.INTER_LINEAR)


    # update metadata
    new_meta = img_meta.copy()
    new_meta['width'] = width
    new_meta['height'] = height
    new_meta['transform'] = img.transform * img.transform.scale(
        (img.width / width),
        (img.height / height)
    )
    new_meta['count'] = channels
    img_meta = new_meta
    
    return img_data, img_meta

def generate_features(img, print_gabor=False):
    """
    Generates features for an image

    Parameters:
    img (numpy.ndarray): The input image.
    print_gabor (bool, optional): Whether to print Gabor filter information. Defaults to False.

    Returns:
    pandas.DataFrame: The DataFrame containing the image data.
    """

    # generate grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # flatten image
    img2 = img.reshape((-1,3))

    # create dataframe
    df = pd.DataFrame(img2, columns=['Blue', 'Green', 'Red'])
    df['Gray'] = gray.reshape(-1)

    # gabor filter
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

    # Roberts Edge
    roberts_edge = roberts(gray).reshape(-1)
    df['Roberts'] = roberts_edge

    # Sobel Edge
    sobel_edge = sobel(gray).reshape(-1)
    df['Sobel'] = sobel_edge

    # Scharr Edge
    scharr_edge = scharr(gray).reshape(-1)
    df['Scharr'] = scharr_edge

    # Prewitt Edge
    prewitt_edge = prewitt(gray).reshape(-1)
    df['Prewitt'] = prewitt_edge

    gaussian_img = nd.gaussian_filter(gray, sigma=3).reshape(-1)
    df['Gaussian s3'] = gaussian_img

    gaussian_img2 = nd.gaussian_filter(gray, sigma=7).reshape(-1)
    df['Gaussian s7'] = gaussian_img2

    median_img =  nd.median_filter(gray, size=3).reshape(-1)
    df['Median s3'] = median_img

    # segmentation
    # felzenszwalb
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    df['Felzenszwalb'] = segments_fz.reshape(-1)

    # quickshift
    segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    df['Quickshift'] = segments_quick.reshape(-1)

    # slic
    segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
    df['SLIC'] = segments_slic.reshape(-1)

    # watershed
    gradient = filters.rank.gradient(gray, disk(2))
    segments_ws = watershed(gradient, markers=250, compactness=0.001)
    df['Watershed'] = segments_ws.reshape(-1)

    return df

def rf_preprocessing(filepath, verbose=False, datatype='uint8', **kwargs):
    """
    Preprocesses an image for random forest classification.
    Image can be resized through the height and width kwargs. It is highly reccomended to scale down the image for time efficiency.
    Reccomended width is <1000.

    Args:
        filepath (str): The path to the image file.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        datatype (str, optional): The desired datatype for the image. Can be 'uint8' or 'float32'. Defaults to 'uint8'.
        **kwargs: Additional keyword arguments for image resizing.

    Returns:
        tuple: A tuple containing the preprocessed image dataframe and the updated image metadata.
    """

    # read kwargs
    if 'height' in kwargs:
        height = kwargs['height']
    else:
        height = None
    if 'width' in kwargs:
        width = kwargs['width']
    else:
        width = None

    # open the image
    img = rasterio.open(filepath)
    img_meta = img.meta

    # read the image
    img_data = img.read()

    # change any nan to 0
    img_data = np.nan_to_num(img_data)

    # fix datatypes
    if datatype == 'uint8':
        img_data = img_data.astype(np.uint8)
    elif datatype == 'float32':
        img_data = (img_data/255.0).astype(np.float32)
    else:
        raise ValueError('Datatype not recognized')

    # check for resizing
    if height is not None or width is not None:
        if height is None:
            height = int(width * img_data.shape[1] / img_data.shape[2])
        if width is None:
            width = int(height * img_data.shape[2] / img_data.shape[1])

        # resize the image
        img_data = cv2.resize(np.transpose(img_data, (1,2,0)), (width, height), interpolation=cv2.INTER_LINEAR)
        img_data = np.transpose(img_data, (2,0,1))

        # update metadata
        new_meta = img_meta.copy()
        new_meta['width'] = width
        new_meta['height'] = height
        new_meta['transform'] = img.transform * img.transform.scale(
            (img.width / width),
            (img.height / height)
        )
        img_meta = new_meta

    # shift shape of img
    img_data = np.transpose(img_data, (1, 2, 0))

    # get image df
    img_df = generate_features(img_data, print_gabor=verbose)

    return img_df, img_meta

def rf_preprocessing_tiled(filepath, verbose=False, datatype='uint8', **kwargs):
    # open the image
    img = rasterio.open(filepath)
    img_meta = img.meta

    # read the image
    img_data = img.read()

    # change any nan to 0
    img_data = np.nan_to_num(img_data)

    # fix datatypes
    if datatype == 'uint8':
        img_data = img_data.astype(np.uint8)
    elif datatype == 'float32':
        img_data = (img_data/255.0).astype(np.float32)
    else:
        raise ValueError('Datatype not recognized')
    
    # break image into tiles
    img_tiles, tile_sizes = break_image(img_data)
    def generator():
        for tile in img_tiles:
            tile = np.transpose(tile, (1,2,0))

            img_df = generate_features(tile, print_gabor=False)

            yield img_df
    
    return generator(), img_meta, tile_sizes


def break_image(image, tile_size = 1024):
    """
    Breaks an image into (512, 512) sections, padding the bottom and right with black pixels.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        list: A list of (512, 512) sections of the image.
    """
    height, width = image.shape[1:]
    padded_height = height + (tile_size - height % tile_size)
    padded_width = width + (tile_size - width % tile_size)

    padded_image = np.zeros((image.shape[0],padded_height, padded_width), dtype=image.dtype)
    padded_image[:,:height, :width] = image

    sections = []
    for i in range(0, padded_height, tile_size):
        for j in range(0, padded_width, tile_size):
            section = padded_image[:, i:i+tile_size, j:j+tile_size]
            sections.append(section)
    return sections, (padded_height//tile_size, padded_width//tile_size)


if __name__=="__main__":
    filepath = "data/2019-20/DeerCreekTrail_2019_05_22_snow.tif"
    rf_preprocessing(filepath, verbose=True)

