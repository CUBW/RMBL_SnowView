import os
import numpy as np

def check_for_pickled_data(filepath):
    """
    Check all .npz files in the given directory for pickled data.
    
    Parameters:
    filepath (str): The directory containing the .npz files.
    
    Returns:
    list: A list of files that contain pickled data.
    """
    pickled_files = []
    all_files = os.listdir(filepath)
    npz_files = [f for f in all_files if f.endswith('.npz')]
    count = 0
    for  npz_file in npz_files:
        try:
            np.load(os.path.join(filepath, npz_file), allow_pickle=False)
            print(count, " file name: ", npz_file, " contains no pickled data")
        except ValueError as e:
            print(f"File {npz_file} contains pickled data: {e}")
            pickled_files.append(npz_file)
        count += 1
    return pickled_files

if __name__ == "__main__":
    FILEPATH = os.path.join(os.path.dirname(__file__), '..', 'data', '512_splits_4_channel.nosync')
    pickled_files = check_for_pickled_data(FILEPATH)
    if pickled_files:
        print("The following files contain pickled data and cannot be loaded with allow_pickle=False:")
        for file in pickled_files:
            print(file)
    else:
        print("No files contain pickled data.")
