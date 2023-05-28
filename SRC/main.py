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

        # execute only once to create the dataset
        # generate_and_dump_dataset(config.root_dir, config.captions_file, config.transforms, cfg.DATA_LOCATION)

        # Generate Dataset
        dataset = make_dataset(config)

        # make the data_loaders, and optimizer
        train_loader, test_loader = make_dataloaders(config, dataset, multiprocessing.cpu_count())

        # Generate vocab
        vocab = dataset.vocab
        config.vocab_size = len(vocab)

        # Get the model
        my_model = make_model(config, device)

        # Make the loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
        optimizer = torch.optim.Adam(my_model.parameters(), lr=config.learning_rate)

        train_loss_arr_epoch = []  # Mean of the losses of the last epoch
        test_loss_arr_epoch = []
        acc_arr_epoch = []

        train_loss_arr_batch = [] # Losses of the batches
        test_loss_arr_batch = []
        acc_arr_batch = []

        train_execution_times = []
        test_execution_times = []

        for epoch in tqdm(range(1, config.epochs + 1)):
            # Training the model
            t0 = time.time()
            train_loss_arr_aux = train(my_model, train_loader, criterion, optimizer, config, epoch)
            t1 = time.time()

            my_model.eval()
            # Testing
            t2 = time.time()
            acc_arr_aux, test_loss_arr_aux = test(my_model, test_loader, criterion, vocab, config, device)
            t3 = time.time()

            # Check how model performs
            test_model_performance(my_model, test_loader, device, vocab, epoch, config)

            my_model.train()

            # Logging data for vizz
            train_loss_arr_epoch.append(sum(train_loss_arr_aux) / len(train_loss_arr_aux))
            test_loss_arr_epoch.append(sum(test_loss_arr_aux) / len(test_loss_arr_aux))

            train_loss_arr_batch += train_loss_arr_aux
            test_loss_arr_batch += test_loss_arr_aux

            acc_arr_epoch.append(sum(acc_arr_aux) / len(acc_arr_aux))
            acc_arr_batch += acc_arr_aux

            train_execution_times.append(t1-t0)
            test_execution_times.append(t3-t2)

        epoch_df = pd.DataFrame([train_loss_arr_epoch, test_loss_arr_epoch, acc_arr_epoch, train_execution_times,
                                 test_execution_times],
                                columns=['epoch_' + str(i) for i in range(len(train_loss_arr_epoch))],
                                index=['train_loss', 'test_loss' ,'test_acc', 'train_times','test_times'])
        loss_batch_df = pd.DataFrame([train_loss_arr_batch],
                                    columns=['batch_' + str(i) for i in range(len(train_loss_arr_batch))],
                                    index=['train_loss'])
        acc_batch_df = pd.DataFrame([acc_arr_batch, test_loss_arr_batch],
                                    columns=['batch_' + str(i) for i in range(len(acc_arr_batch))],
                                    index=['test_acc', 'test_loss'])

        if config.save:
            epoch_df.to_csv(config.DATA_LOCATION+'/logs'+'/epoch_df.csv')
            loss_batch_df.to_csv(config.DATA_LOCATION+'/logs'+'/loss_batch_df.csv')
            acc_batch_df.to_csv(config.DATA_LOCATION+'/logs'+'/acc_batch_df.csv')
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
        root_dir=DATA_LOCATION+"/Images",
        captions_file=DATA_LOCATION+"/captions.txt",
        device=device,
        encoder='ResNet50',
        transforms=transforms,
        embed_size=300,
        attention_dim=256,
        encoder_dim=2048,
        decoder_dim=512,
        epochs=25,
        learning_rate=3e-4,
        batch_size=int(256),
        DATA_LOCATION=DATA_LOCATION,
        train_size=0.1,
        save=True
    )

    model = model_pipeline(config)
