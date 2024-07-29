import os
import glob
import numpy as np
import time
import pickle

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SOURCE = "data/512_splits_4_channel/"
OUTPUT_DIR = "3_channel_results/"
TRAIN_BATCHES = 20

def load_data_in_batches(img_filenames):
    for filename in img_filenames:
        # break name up into tags
        mask_name = filename.replace("images", "masks")
        
        # open files
        img_data = np.load(filename)
        mask_data = np.load(mask_name)
        
        # loop through arrays and process
        img_arrays = img_data.files
        mask_arrays = img_data.files
        
        # process data in chunks to limit number of array copying
        chunk = np.empty((0,4), dtype=np.uint8)
        chunk_size = 300 # number is a total guess but seems to work, each batch is ~ 3900 tiles
        
        for i in range(len(img_arrays)):
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
                print(f"Chunk of size{chunk.shape[0]} loaded.")
                yield chunk
                # reset chunk
                chunk = np.empty((0,4), dtype=np.uint8)
        img_data.close()
        mask_data.close()
    
def train_in_batches(img_filenames, val_filenames, estimator_per_batch = 1):
    params = {
        'criterion': 'entropy',
        'max_depth': 20,
        'max_features': 'log2',
        'min_samples_leaf': 4,
        'min_samples_split': 5,
    }
    rf = RandomForestClassifier(warm_start=True, n_estimators=estimator_per_batch, verbose=2, n_jobs=-1)
    print("Beginning training in batches.")
    
    # history = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[]}
    
    # loop through batches
    n_estimators = 0
    for data in load_data_in_batches(img_filenames):
        # update estimators
        n_estimators+=estimator_per_batch
        rf.set_params(n_estimators = n_estimators)
        
        start_time = time.time()
        
        # train model
        rf.fit(data[:,:-1], data[:,-1])
        
        # # evaluate model
        # metrics = evaluate_model_in_batches(rf, val_filenames)
        # history['accuracy'].append(history['accuracy'])
        # history['precision'].append(metrics['precision'])
        # history['recall'].append(metrics['recall'])
        # history['f1'].append(metrics['f1'])
        
        print(f"Trained {estimator_per_batch} more trees ({n_estimators} total) in {time.time()-start_time} seconds.")
        
        # save model
        with open(os.path.join(OUTPUT_DIR, 'checkpoints', f'3_feature_checkpoint.pkl'),'wb') as f:
            pickle.dump(rf, f)
        # print(metrics)

        
    print("Completed training.")
    
    return rf
    
def evaluate_model_in_batches(model, filepaths):
    print("Beginning evaluation")
    
    #loop through batches
    metrics = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[]}
    for data in load_data_in_batches(filepaths):
        
        # predict 
        pred_y = model.predict(data[:,:-1])
        
        # get metrics and store
        metrics['accuracy'].append(accuracy_score(data[:,-1], pred_y))
        metrics['precision'].append(precision_score(data[:,-1], pred_y))
        metrics['recall'].append(recall_score(data[:,-1], pred_y))
        metrics['f1'].append(f1_score(data[:,-1], pred_y))

    # average metrics
    avg_metrics = {k:np.mean(v) for k,v in metrics.items()}
    
    return avg_metrics
        
        
    
def main():
    #create output dierctory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)
    
    
    start_time = time.time()
    
    # generate filenames
    img_filepaths = glob.glob(os.path.join(SOURCE, 'train*images.npz'), recursive=True)
    val_filepaths = glob.glob(os.path.join(SOURCE, 'val*images.npz'), recursive=True)
    
    # train model
    rf = train_in_batches(img_filenames=img_filepaths, val_filenames= val_filepaths)
    
    # save model
    with open(os.path.join(OUTPUT_DIR, "3_feature_model.pkl"), 'wb') as f:
        pickle.dump(rf, f)
        
    # # save history
    # with open(os.path.join(OUTPUT_DIR, "history.txt"), 'w') as f:
    #     f.write(str(history))
        
    # evaluate model
    
    # generate filenames
    test_filepaths = glob.glob(os.path.join(SOURCE, 'test*images.npz'), recursive=True)
    
    # get metrics
    test_metrics = evaluate_model_in_batches(rf, test_filepaths)
    
    # save metrics to file
    with open(os.path.join(OUTPUT_DIR, "results.txt"), 'w') as file:
        for key, values in test_metrics.items():
            file.write(f'{key}: {values}\n')
        
        file.write(f'Total time: {time.time()-start_time} seconds')
    

if __name__ =="__main__":
    main()