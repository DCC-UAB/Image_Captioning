import os
import random
import wandb

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from train import *
from test import *
from utils.utils import *
from models.models import *

# Global variables
global global_vocab
global device

# Setting CUDA ALLOC split size to 256 to avoid running out of memory
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def model_pipeline(cfg: dict):
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=cfg):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # execute only once to create the dataset
        # generate_and_dump_dataset(config.root_dir, config.captions_file, config.transforms, cfg.DATA_LOCATION)

        # make the data_loaders, and optimizer
        train_loader, test_loader = make_dataloaders(config)

        # Sets vocab and returns vocab size
        config.vocab_size = get_vocab_size(config)

        # Get the model
        my_model, criterion, optimizer = make_model(config, device)

        # and use them to train the model
        train(my_model, train_loader, criterion, optimizer, config)

        # and test its final performance
        test(my_model, test_loader, global_vocab)

    return my_model


if __name__ == "__main__":
    wandb.login()

    transforms = T.Compose([
        T.Resize(226),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    DATA_LOCATION = '../data'

    config = dict(
        root_dir=DATA_LOCATION+"/Images",
        captions_file=DATA_LOCATION+"/captions.txt",
        transforms=transforms,
        embed_size=300,
        attention_dim=256,
        encoder_dim=2048,
        decoder_dim=512,
        epochs=25,
        learning_rate=3e-4,
        batch_size=256,
        DATA_LOCATION=DATA_LOCATION)

    model = model_pipeline(config)
