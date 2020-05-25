import sys
sys.path.append('../')
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from config import DATA_PATH
import pdb


class fashionNet(Dataset):
    def __init__(self, subset):
        """Dataset class representing fashionNet dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.CenterCrop(96),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            fashionNet dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/fashionNet/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.jpg')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/fashionNet/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images