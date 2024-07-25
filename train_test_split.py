import os
import glob
import numpy as np
import random
import time

SOURCE = "output_4_channel"
OUTPUT_DIRECTORY = "data/512_splits"

def generate_indicies_list(files):
    all_indices = []
    for filename in files:
        data = np.load(filename)
        num = len(data.files)
        all_indices = all_indices + [(filename, i) for i in range(num)]
        data.close()

    return all_indices

def split_indices(indices):
    random.seed(42)
    random.shuffle(indices)

    split_70 = int(0.7 * len(indices))
    split_20 = int(0.2 * len(indices))

    train_indices = indices[:split_70]
    val_indices = indices[split_70:split_70+split_20]
    test_indices = indices[split_70+split_20:]

    print(f"Train: {len(train_indices)}")
    print(f"Validation: {len(val_indices)}")
    print(f"Test: {len(test_indices)}")

    # sort each by their filename
    train_indices.sort(key=lambda x: x[0])
    val_indices.sort(key=lambda x: x[0])
    test_indices.sort(key=lambda x: x[0])

    return train_indices, val_indices, test_indices

def generate_files(indices, type, output_dir = OUTPUT_DIRECTORY, num_files = 20):

    indices_per_file = len(indices)//num_files

    #turn train_indices into a dictionary filename:[indicies]
    indices_dict = {}
    for filename, index in indices:
        if filename not in indices_dict:
            indices_dict[filename] = []
        indices_dict[filename].append(index)

    current_file = 0
    current_image_output = []
    current_mask_output = []

    # loop through all filenames, then indicies in those files and add them to bigger file
    for filename, file_indices in indices_dict.items():
        # get mask_filename
        mask_filename = filename.replace("images", "masks")

        image_data = np.load(filename) # open image
        mask_data = np.load(mask_filename) # open mask

        image_array_names = image_data.files
        mask_array_names = mask_data.files

        for index in file_indices:
            current_image_output.append(image_data[image_array_names[index]])
            current_mask_output.append(mask_data[mask_array_names[index]])

            if len(current_image_output) == indices_per_file and current_file < num_files-1:
                np.savez_compressed(os.path.join(output_dir, f"{type}_{current_file}_images"), *current_image_output)
                np.savez_compressed(os.path.join(output_dir, f"{type}_{current_file}_masks"), *current_mask_output)
                current_image_output = []
                current_mask_output = []
                print(f"Saved {current_file}")
                current_file += 1
        image_data.close()
        mask_data.close()

    # save last file 
    np.savez_compressed(os.path.join(output_dir, f"{type}_{current_file}_images"), *current_image_output)
    np.savez_compressed(os.path.join(output_dir, f"{type}_{current_file}_masks"), *current_mask_output)

def format_seconds(seconds):
    """
    Convert a float number of seconds into a string formatted as HH:MM:SS, MM:SS, or SS.##.
    
    Args:
    seconds (float): The number of seconds to format.
    
    Returns:
    str: The formatted time string.
    """
    if seconds >= 3600:
        # More than an hour
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02}:{minutes:02}:{secs:05.2f}"
    
    elif seconds >= 60:
        # More than a minute but less than an hour
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02}:{secs:05.2f}"
    
    else:
        # Less than a minute
        return f"{seconds:05.2f}"
    
if __name__ =="__main__":
    og_start_time = time.time()
    # create ouput directory
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # get all files in the directory
    files = glob.glob(os.path.join(SOURCE, '**', '*_images.npz'), recursive=True)

    indices = generate_indicies_list(files)

    train_indices, val_indices, test_indices = split_indices(indices)

    # write indices to file
    with open(os.path.join(OUTPUT_DIRECTORY, "split_indices.txt"), "w", newline='') as f:
        f.write("filename,index,split\n")
        for filename, index in train_indices:
            f.write(f"{filename},{index},train\n")
        for filename, index in val_indices:
            f.write(f"{filename},{index},val\n")
        for filename, index in test_indices:
            f.write(f"{filename},{index},test\n")
    f.close()

    print(f"Preparation time: {format_seconds(time.time() - og_start_time)}")
    start_time = time.time()

    # actually generate split files
    print("Generating  train files")
    generate_files(train_indices, "train", num_files=20)
    print(f"Train generation time: {format_seconds(time.time() - start_time)}")
    start_time = time.time()

    print("Generating val files")
    generate_files(val_indices, "val", num_files=5)
    print(f"Val generation time: {format_seconds(time.time() - start_time)}")
    start_time = time.time()

    print("Generating test files")
    generate_files(test_indices, "test", num_files=3)
    print(f"Test generation time: {format_seconds(time.time() - start_time)}")

    print(f"Completed split. Results stored in {OUTPUT_DIRECTORY}")
    print(f"Total time: {format_seconds(time.time() - og_start_time)}")
