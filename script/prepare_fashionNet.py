"""
Run this script to prepare the Fashion dataset.

1. Download files from https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1 and place in
    data/fashionNet/
2. Run the script

"""

import sys
sys.path.append('../')
import os
import shutil
import natsort
import numpy as np
from tqdm import tqdm as tqdm
import pandas as pd
from config import DATA_PATH
from few_shot.utils import mkdir, rmdir

# Creating and deleting folder/files
rmdir(DATA_PATH + '/fashionNet/images_background')
rmdir(DATA_PATH + '/fashionNet/images_evaluation')
rmdir(DATA_PATH + '/fashionNet/refac_Images')
mkdir(DATA_PATH + '/fashionNet/images_background')
mkdir(DATA_PATH + '/fashionNet/images_evaluation')
mkdir(DATA_PATH + '/fashionNet/refac_Images')

print("Is the DATA_PATH is Correct?", os.path.exists(DATA_PATH + '/fashionNet/images/'))

'''
Directory File Name Change
1. styles.csv to map image_id and subCategory, class_laebls, meta_sets
2. Rename the images using os.rename() for support and query split

'''
_classes = []
PATH = os.path.join(DATA_PATH, 'fashionNet/styles.csv')
df = pd.read_csv(PATH, engine='python')
df = df.drop(['Unnamed: 10', 'Unnamed: 11'], axis=1)
df = df.sort_values(by=['id'], ascending=True, inplace=False)
df['class_id'] = df.apply(lambda row: str(row['subCategory']) + '__' + str(row['articleType']) + '__' +  str(row['id']) + '.jpg', axis=1)
for name in df['class_id']:
	_classes.append(name)

'''
_classes: ['Topwear__Tshirts__1163.jpg', 'Topwear__Tshirts__1164.jpg', 'Topwear__Tshirts__1165.jpg', 'Bags__Backpacks__1525.jpg', 
		'Bags__Backpacks__1526.jpg']
Format: [ subCategory__articleType__image_id]

'''
for root, _, files in os.walk(DATA_PATH + '/fashionNet/images/'):
	dirFiles = natsort.natsorted(files, reverse=False)
	for name in range(len(_classes)):
		src_dir = os.path.join(DATA_PATH + '/fashionNet/images/', dirFiles[name])
		dst_dir = DATA_PATH + '/fashionNet/refac_Images/{}'.format(_classes[name])
		os.rename(src_dir, dst_dir)

# Find class identities
classes = []
for root, _, files in os.walk(DATA_PATH + '/fashionNet/refac_Images/'):
    for f in files:
        if f.endswith('.jpg'):
        	f = f.split('__')
        	classes.append(f[0]+'__'+ f[1])

class_name = list(set(classes))
# print(len(class_name))

# Meta Training and Testing Split (Support and Query set)
meta_train_PATH = DATA_PATH + '/fashionNet/Meta/meta_train.csv'
meta_train_class = []
df = pd.read_csv(meta_train_PATH, engine='python')
for items in df['meta_train']:
	meta_train_class.append(items)

meta_test_PATH = DATA_PATH + '/fashionNet/Meta/meta_test.csv'
meta_test_class = []
df = pd.read_csv(meta_test_PATH, engine='python')
for items in df['meta_test']:
	meta_test_class.append(items)

# Meta Train/Test Split (based on Meta Training and Testing set)
background_classes = []
for item in meta_train_class:
	for substring in class_name:
		if item in substring:
			background_classes.append(substring)

evaluation_classes = []
for item in meta_test_class:
	for substring in class_name:
		if item in substring:
			evaluation_classes.append(substring)

# Create class folders
for c in background_classes:
    mkdir(DATA_PATH + '/fashionNet/images_background/{}/'.format(c))
 
for c in evaluation_classes:
    mkdir(DATA_PATH + '/fashionNet/images_evaluation/{}/'.format(c))

# Move images to correct location
for root, _, files in os.walk(DATA_PATH + '/fashionNet/refac_Images'):
    for f in tqdm(files, total=len(files)):
    	if f.endswith('.jpg'):
            name = f.split('__')
            class_name = name[0] + '__' + name[1]
            image_name = name[0] + '__' + name[1] + '__' + name[2]
            # Send to correct folder
            if class_name not in evaluation_classes and background_classes:
            	continue
            subset_folder = 'images_background' if class_name in background_classes not in evaluation_classes else 'images_evaluation' 
            src = '{}/{}'.format(root, f)
            dst = DATA_PATH + '/fashionNet/{}/{}/{}'.format(subset_folder, class_name, image_name)
            shutil.copy(src, dst) #Time Complexity O(n), for n num_of samples	