#imports
import torchvision.transforms as T
import multiprocessing
import joblib

#import dataloader
from utils.utils import *

# Defining the transform to be applied
transforms = T.Compose([
    T.Resize(226),
    T.RandomCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

# Initialize the dataset class
dataset =  FlickrDataset(
    root_dir = DATA_LOCATION+"/Images",
    captions_file = DATA_LOCATION+"/captions.txt",
    transform=transforms
)

joblib.dump(dataset, DATA_LOCATION+"/processed_dataset.joblib")