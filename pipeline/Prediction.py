import pickle
import os
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier 
from tqdm import tqdm
# TODO imports for nn models

from Preprocessing import nn_preprocessing, rf_preprocessing, rf_preprocessing_tiled
from Postprocessing import save_image, display_both, combine_tiles

IMG_FILEPATH = 'C:/Users/apfox/UCB-O365/Travis Hainsworth - RMBL/2 - Source Data/2019-2020_Data/Imagery/DeerCreekTrail/DeerCreekTrail_2019_05_22_snow.tif' # change this filepath if running Prediciton.py
OUTPUT_DIRECTORY = "./predictions/" # change this directory if running 

def predict_nn(model, img):
    """
    Predicts the output of the neural network model for the given image.

    Parameters:
        model (object): The trained neural network model.
        img (object): The preprocessed input image for prediction.

    Returns:
        object: The predicted output of the model.
    """
    pass # TODO

def predict_rf(model, img_df, verbose = True):
    """
    Predicts the output of the random forest model for the given image.

    Parameters:
        model (object): The trained random forest model.
        img (object): The preprocessed input image for prediction.
        verbose (bool, optional): Whether to display additional information during the prediction. Defaults to True.

    Returns:
        object: The predicted output of the model.
    """
    if not verbose:
        model.set_params(verbose=0)
    else:
        model.set_params(verbose=1)

    return model.predict(img_df)

def full_pipeline(img_filepath, model_type, output_dir, verbose=False, tile=False, **kwargs):
    """
    Runs the full pipeline for image prediction given an image and a model.
    It is highly reccomended to scale down the image if using a Random Forest model for time efficiency.
    Reccomended width is <1000

    Args:
        img_filepath (str): The file path of the input image.
        model_type (str): The type of the model to use for prediction. Must be one of ['RF', 'U-NET', 'DEEPLAB'].
        output_dir (str): The directory to save the predicted image.
        verbose (bool, optional): Whether to display additional information during the pipeline. Defaults to False.
        **kwargs: Additional keyword arguments specific to the preprocessing functions.

    Raises:
        ValueError: If an invalid model type is provided.
        FileNotFoundError: If the model file or input image file is not found.
        ValueError: If the input image is not in GeoTIFF format.

    Returns:
        None
    """
    # check model_type
    if model_type not in ['RF', 'U-NET', 'DEEPLAB']:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # confirm model exists
    model_filepath = f"./models/{model_type}.pkl"
    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"Model not found at '{model_filepath}'")

    # load in model
    with open(model_filepath, 'rb') as f:
        model = pickle.load(f)

    
    
    # confirm image exists and is a tif
    if not os.path.exists(img_filepath):
        raise FileNotFoundError(f"Image not found at '{img_filepath}'")
    if not img_filepath.endswith('.tif'):
        raise ValueError(f"Image must be a .tif file(in GeoTIFF format)")

    if not tile:

        # preprocess image
        if model_type == 'RF':
            img, img_meta = rf_preprocessing(img_filepath, verbose=verbose, **kwargs)
        else:
            img, img_meta = nn_preprocessing(img_filepath, **kwargs)

        # check it is GeoTIFF
        if img_meta['driver'] != 'GTiff':
            raise ValueError("Image must be in GeoTIFF format")
        
        # predict with choosen model
        if model_type == 'RF':
            # get prediciton
            pred = predict_rf(model, img, verbose=verbose)

            # reshape prediction
            pred = pred.reshape((img_meta['height'], img_meta['width'], 1))
            pred = pred.transpose((2, 0, 1))
        else:
            pred = predict_nn(model, img)

        # update metadata to one channel
        img_meta['count'] = 1

        # make output_filepath
        basename = os.path.basename(img_filepath)
        output_filepath = os.path.join(output_dir, f"{basename}_{model_type}_prediction.tif")

        # save image
        save_image(pred, img_meta, output_filepath)

        if verbose:
            # get the red, green and blue columns and reshape
            img_rgb = img[['Blue', 'Green', 'Red']].values.reshape((img_meta['height'], img_meta['width'], 3))

            # display the image and the mask
            display_both(img_rgb, pred[0])
    
    else: 
        if model_type == 'RF':
            tiles = []
            generator, img_meta, tile_sizes = rf_preprocessing_tiled(img_filepath, verbose=verbose)
            for tile in tqdm(generator, unit='tile'):
                # get prediciton
                pred = predict_rf(model, tile, verbose=verbose)

                # reshape prediction
                pred = pred.reshape((1024, 1024, 1))
                pred = pred.transpose((2, 0, 1))

                # add prediction to output list
                tiles.append(pred)
            
            # combine tiles

            # make output_filepath
            basename = os.path.basename(img_filepath)
            output_filepath = os.path.join(output_dir, f"{basename}_{model_type}_prediction.tif")

            img_meta['count'] = 1
            with open('tiles.pkl', 'wb') as f:
                pickle.dump((tiles, img_meta, tile_sizes, output_filepath), f)
            pred_img = combine_tiles(tiles, img_meta, tile_sizes, output_filepath)

            if verbose:
                #display mask with maptpltlib
                plt.imshow(pred_img[0], cmap='gray')
                plt.show() 
    

if __name__ == '__main__':
    # make output_dir
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.mkdir(OUTPUT_DIRECTORY)

    # run full pipeline
    full_pipeline(IMG_FILEPATH, 'RF', OUTPUT_DIRECTORY, verbose=True, tile=True)
    # full_pipeline(IMG_FILEPATH, 'U-NET', OUTPUT_DIRECTORY)
