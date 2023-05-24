#import dataloader
from utils.utils import *

#imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import multiprocessing

import joblib



#setting the constants
data_location =  "./data"
BATCH_SIZE = 256
NUM_WORKER = multiprocessing.cpu_count()

#defining the transform to be applied
transforms = T.Compose([
    T.Resize(226),
    T.RandomCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

#initialize the dataset class
dataset =  FlickrDataset(
    root_dir = data_location+"/Images",
    captions_file = data_location+"/captions.txt",
    transform=transforms
)

joblib.dump(dataset, "./processed_dataset.joblib")