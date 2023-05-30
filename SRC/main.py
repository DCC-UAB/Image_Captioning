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
import multiprocessing



# Global variables
global device

import os

# Setting CUDA ALLOC split size to 256 to avoid running out of memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# Stopping wandb from creating symlinks
os.environ["WANDB_DISABLE_SYMLINKS"] = "true"

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

        # Execute only once to create the dataset
        # generate_and_dump_dataset(config.root_dir, config.captions_file, config.transforms, cfg.DATA_LOCATION)

        # Generate Dataset
        dataset = make_dataset(config)

        # Get the data loaders
        train_loader, test_loader = make_dataloaders(config, dataset, 1)

        # Generate vocab
        vocab = dataset.vocab
        config.vocab_size = len(vocab)

        # Get the model
        my_model = make_model(config, device)

        # Define the loss and optimizer
        criterion = get_criterion(config.criterion, vocab.stoi["<PAD>"])
        criterion.ignore_index=vocab.stoi["<PAD>"]
        
        optimizer = get_optimizer(config.optimizer, my_model.parameters(), config.learning_rate)
        
        # Arrays to log data
        train_loss_arr_epoch, test_loss_arr_epoch, acc_arr_epoch  = [], [], [] # Epoch-wise
        train_loss_arr_batch, test_loss_arr_batch, acc_arr_batch = [], [], [] # Batch-wise
        train_execution_times, test_execution_times = [], [] # Execution times

        
        for epoch in tqdm(range(1, config.epochs + 1)):
            # Training
            my_model.train()
            train_loss_arr_aux, train_time = train(my_model, train_loader, criterion, optimizer, config, epoch)
            my_model.eval()

            # Testing
            acc_arr_aux, test_loss_arr_aux, test_time = test(my_model, test_loader, criterion, vocab, config, device)

            # Check how model performs
            test_model_performance(my_model, test_loader, device, vocab, epoch, config)
            
            # Logging data for vizz
            train_loss_arr_epoch.append(np.mean(train_loss_arr_aux)); test_loss_arr_epoch.append(np.mean(test_loss_arr_aux))
            train_loss_arr_batch += train_loss_arr_aux; test_loss_arr_batch += test_loss_arr_aux
            acc_arr_epoch.append(np.mean(acc_arr_aux)); acc_arr_batch += acc_arr_aux
            train_execution_times.append(train_time); test_execution_times.append(test_time)

            
        if config.save:
            export_data(train_loss_arr_epoch, test_loss_arr_epoch, acc_arr_epoch, train_execution_times, test_execution_times,
                   train_loss_arr_batch, acc_arr_batch, test_loss_arr_batch, config)
            
            save_model(my_model, config, config.DATA_LOCATION+'/logs'+'/EncoderDecorder_model.pth')

    return my_model


if __name__ == "__main__":
    wandb.login()

    print("Using: ", device)

    transforms = T.Compose([
        T.Resize(226),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    DATA_LOCATION = '../data'

    config = dict(
        # Paths
        root_dir=DATA_LOCATION+"/Images",
        captions_file=DATA_LOCATION+"/captions.txt",
        DATA_LOCATION=DATA_LOCATION,
        save=True,

        # Training data
        epochs=1,
        batch_size=50,
        train_size=0.1,
        
        # Model data
        optimizer='Adam',
        criterion='CrossEntropy',
        learning_rate=0.0001,
        device=device,
        encoder='ResNet50',
        transforms=transforms,
        embed_size=300,
        attention_dim=256,
        encoder_dim=2048,
        decoder_dim=512,
    )

    model = model_pipeline(config)
