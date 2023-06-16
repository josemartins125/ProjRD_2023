import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
import shutil
import re
from multiprocessing import Pool
from tqdm import tqdm


def wipe_data(data_directory, final_run=True):
    if final_run:
        # Specify the folders for the final run
        train_directory = os.path.join(data_directory, "train")
        val_directory = os.path.join(data_directory, "validation")

        folders = [
            os.path.join(train_directory, "0"),
            os.path.join(train_directory, "1"),
            os.path.join(val_directory, "0"),
            os.path.join(val_directory, "1")
        ]
    else:
        # Specify the folders for the test run
        folders = [
            os.path.join(data_directory, "0"),
            os.path.join(data_directory, "1")
        ]

    # Create parent directories if they don't exist
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # Display progress bar
    with tqdm(total=len(folders), desc="Deleting data") as pbar:
        for folder in folders:
            # Delete all files within the folder
            for file_name in os.listdir(folder):
                file_path = os.path.join(folder, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            pbar.update(1)

    print("Data wiped successfully.")


def manual_select_new(df):
    c_B = 0
    c_M = 0

    # Create boolean arrays for Benign and Malicious labels
    M_B = (df['label'] == 'Benign').astype(int)
    M_M = (df['label'] == 'Malicious').astype(int)

    # Count the occurrences
    c_B = M_B.sum()
    c_M = M_M.sum()

    print('Benign=', c_B, '')
    print('Malicious=', c_M, '\n')
    return M_B, M_M


def process_row(row):
    adstr = ''.join(row.astype(str))
    img = Image.new('L', (850, 11))
    d = ImageDraw.Draw(img)
    d.text((0, 0), adstr, 255)
    return img


def save_image(args):
    img, M_M, index, directory = args
    label = str(M_M[index])
    
    img.save(os.path.join(directory, label, "{}_{}.jpg".format(label, index)), "JPEG")


def write_data(df, M_M, directory):
    print("Writing data...") 
    # Convert M_M to NumPy array
    M_M = M_M.to_numpy()

    # Use multiprocessing for parallel image saving
    pool = Pool()

    # Prepare arguments for image saving
    args = [(process_row(row), M_M, index, directory) for index, row in df.iterrows()]

    # Save images in parallel with progress bar
    with tqdm(total=len(args), desc="Saving images") as pbar:
        for _ in pool.imap_unordered(save_image, args):
            pbar.update(1)

    pool.close()
    pool.join()

    print("Writing data complete")




def shuffle_and_truncate(df, size):
    df = df.sample(frac=1).reset_index(drop=True)
    df = df[:size]
    return df


if __name__ == "__main__":
    datasets = ["iot23_dataset_1.csv", "iot23_dataset_2.csv","iot23_dataset_3.csv"]
    for count, dataset in enumerate(datasets):
        df = pd.read_csv(dataset)
        df = shuffle_and_truncate(df, size=100000)

        M_B, M_M = manual_select_new(df)

        # Split data into train and validation sets
        split_point = round(len(df) * 0.8)
        train_df = df[:split_point]
        val_df = df[split_point:]

        # Specify the directory where the data is stored
        data_directory = f"./data/train_images{count}"
        train_directory = os.path.join(data_directory, "train")
        val_directory = os.path.join(data_directory, "validation")

        wipe_data(data_directory, final_run=True)

        write_data(train_df, M_M, train_directory)
        write_data(val_df, M_M, val_directory)