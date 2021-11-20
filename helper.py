import sys
import os
import shutil
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import random
from multiprocess import Pool

def show_random_images(count=10):
    fig = plt.subplots(figsize=(18, count/2*5))
    ncols, nrows = 5, int(count/2)+1
    
    for i in range(1, count+1):
        plt.subplot(nrows, ncols, i)
        image_dirs = os.listdir('./images')
        random_dir = random.choice(image_dirs)
        images = os.listdir(f'./images/{random_dir}')
        random_image = random.choice(images)

        img = PIL.Image.open(f'./images/{random_dir}/{random_image}')
        plt.title(f'Image Class: {random_dir}', loc='left', fontsize=10)
        plt.axis('off')
        plt.imshow(img)
        
    plt.show()


def generate_train_test_from_folder(img_path: str, train_size: float) -> dict:
    """
    Performs a per class image splits by given trainsize in pct.
    
    params:
    -----------
    img_path: str
        path to image folder with structure: {img_path}/{*class_names}/img.jpg
    
    train_size: float
        Train test ratio to split data-
        
    returns:
    -----------
    img_splits: dict
        dictionary with per class train test splits
        dict-obj like:{class_name1: {train: [img1.jpg, ...],
                                    test: [img234.jpg. ...]},
                       class_name2: {...}
                       ...}
        
    """
    img_classes = os.listdir(img_path)
    
    img_splits = {}
    for cls in tqdm(img_classes):
        img_splits[cls] = {}
        cls_images = os.listdir(f'{img_path}/{cls}')
        cls_images = np.array(cls_images) 
        
        # Get sizes per class
        amount_train_images = int(len(cls_images) * train_size)
        
        # Perform split
        cls_images = np.random.permutation(cls_images)
        train_images = cls_images[:amount_train_images]
        test_images= cls_images[amount_train_images:]
        img_splits[cls]['train'] = ['/'.join([img_path,cls,img]) for img in train_images]
        img_splits[cls]['test'] = ['/'.join([img_path,cls,img]) for img in test_images]
        
    return img_splits

def create_train_test_folder(split_dict: dict, fp_new_folder: str):
    """
    Creates New Folder structure within given fp_new_folder.
    """
    train_path = fp_new_folder + '/' + 'train' 
    test_path = fp_new_folder + '/' + 'test'
    try:
        os.mkdir(fp_new_folder)
    except Exception as e:
        print(e)
    try:
        for fp in [train_path, test_path]:
            os.mkdir(fp)
    except Exception as e:
        print(e)
    try:
        for cls in list(split_dict.keys()):
            os.mkdir(train_path + '/' + cls)
            os.mkdir(test_path + '/' + cls)
    except Exception as e:
        print(e)
    
def prepare_img_splits(split_dict: dict, new_fp: str) -> list:
    """
    Prepares the Dictionary obtained by function generate_train_test_from_folder into list 
    which can be used with multiprocessing.
    
    params:
    -----------
    split_dict: dict
        dictionary generated wiht generate_train_test_from_folder() function
        which splits data in train and testset
    
    new_fp: str
        new Filepath, where the new folder structure with train test should
        be created
    
    returns:
    -----------
    None
    
    """
    # Create list
    file_path_from_to = []
    for cls in tqdm(split_dict.keys(), desc=' Creating iterable...'):
        for split, old_paths in split_dict[cls].items():
            for p in old_paths:
                new_path_train = f'{new_fp}/{split}' + '/' + '/'.join(p.split('/')[1:])
                file_path_from_to.append(p + '__' + new_path_train)          
    
    return file_path_from_to
        
def copy_all_files(fp_from_to: list, n_proc=4):
    """Wrapper Function for Multiprocessing"""
    # Copy all Files
    with Pool(processes=n_proc) as pool:
        for _ in tqdm(pool.imap_unordered(_copy_to, fp_from_to), total=len(fp_from_to)):
            pass

def _copy_to(files_string):
    """Function for Multiprocessing"""
    from shutil import copyfile ## Questionable workaround??
    file_from, file_to = files_string.split('__')
    _ = copyfile(file_from, file_to)
    
    
def drop_corrupted_images(image_paths: list):
    """
    As I wanted to fit a neuronal Network I recognized that I have corrupted images in my data.
    Like imagefiles without any content. I wrote this function to remove them.
    
    First counts corrutped images and ask if should be removed.
    
    params:
    ------------
    image_pahts: list
        paths generated with function prepare_img_splits()
        
    returns:
    ------------
    None
    
    """
    image_paths = [path.split('__')[0] for path in image_paths]
    images_to_drop = []
    for path in tqdm(image_paths, desc='Search for Corrupted Images ... '):
        try:
            Image.open(path)
        except:
            images_to_drop.append(path)
    
    print(f'Found {len(images_to_drop)} corrupted images out of {len(image_paths)}')
    
    if input('Remove Images? (y/n)') == 'y':
        for path in tqdm(images_to_drop, desc='Removing Images ... '):
            try:
                os.remove(path)
            except FileNotFoundError as e:
                print(e)