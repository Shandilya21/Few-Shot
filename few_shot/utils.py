import torch
import os
import sys
sys.path.append('../')
import shutil
from typing import Tuple, List

from config import EPSILON, PATH


def mkdir(dir):
    """Create a directory, ignoring exceptions

    # Arguments:
        dir: Path of directory to create
    """
    try:
        os.mkdir(dir)
    except:
        pass


def rmdir(dir):
    """Recursively remove a directory and contents, ignoring exceptions

   # Arguments:
       dir: Path of directory to recursively remove
   """
    try:
        shutil.rmtree(dir)
    except:
        pass


def setup_dirs():
    """Creates directories for this project."""
    mkdir(PATH + '/logs/')
    mkdir(PATH + '/logs/proto_nets')
    mkdir(PATH + '/logs/matching_nets')
    mkdir(PATH + '/logs/maml')
    mkdir(PATH + '/models/')
    mkdir(PATH + '/models/proto_nets')
    mkdir(PATH + '/models/matching_nets')
    mkdir(PATH + '/models/maml')


def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.

    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
        
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))