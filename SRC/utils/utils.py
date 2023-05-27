import pandas as pd
from collections import Counter
import spacy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import joblib
from copy import deepcopy
from sklearn.model_selection import train_test_split
from SRC.models.models import *
import time


def flickr_train_test_split(dataset, train_size, test_size):
    # Splits dataset with same vocabulary
    train_X, test_X = train_test_split(dataset.df, train_size=train_size, test_size=test_size)

    train_dataset = deepcopy(dataset)
    test_dataset = deepcopy(dataset)

    train_dataset.df = train_X
    train_dataset.df.reset_index(drop=True, inplace=True)
    train_dataset.imgs = train_X['image']
    train_dataset.captions = train_X['caption']

    test_dataset.df = test_X
    test_dataset.df.reset_index(drop=True, inplace=True)
    test_dataset.imgs = test_X['image']
    test_dataset.captions = test_X['caption']

    return train_dataset, test_dataset

def make_dataset(config):
    dataset = joblib.load(config.DATA_LOCATION + "/processed_dataset.joblib")
    return dataset

# Make initializations
def make_dataloaders(config, dataset):
    train_dataset, test_dataset = flickr_train_test_split(dataset, config.train_size, config.test_size)

    train_loader = get_data_loader(train_dataset, batch_size=config.batch_size)
    test_loader = get_data_loader(test_dataset, batch_size=config.test_batch_size)

    return train_loader, test_loader


def make_model(config, device='cuda'):
    # make the model
    model = EncoderDecoder(config.embed_size, config.vocab_size, config.attention_dim, config.encoder_dim,
                           config.decoder_dim, device=device).to(device)

    return model


class Vocabulary:
    def __init__(self, freq_threshold):
        # setting the pre-reserved tokens int to string tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

        # string to int tokens
        # its reverse dict self.itos
        self.stoi = {v: k for k, v in self.itos.items()}

        self.freq_threshold = freq_threshold
    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text, spacy_eng):

        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        i = 1
        length = len(sentence_list)
        for sentence in sentence_list:
            i += 1
            if not (i % 100):
                print(i, "Iterations done out of", length)
            for word in self.tokenize(sentence, self.spacy_eng):
                frequencies[word] += 1

                # add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text, spacy_eng):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text, spacy_eng)

        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]


class FlickrDataset(Dataset):
    """
    FlickrDataset
    """

    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Get image and caption column from the dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())
        self.spacy_eng = spacy.load("en_core_web_sm")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir, img_name)
        img = Image.open(img_location).convert("RGB")

        # apply the transformation to the image
        if self.transform is not None:
            img = self.transform(img)

        # numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption, self.spacy_eng)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(caption_vec)


class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """

    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs, targets


def get_data_loader(dataset, batch_size, shuffle=False, num_workers=1):
    """
    Returns torch dataloader for the flicker8k dataset
    
    Parameters
    -----------
    dataset: FlickrDataset
        custom torchdataset named FlickrDataset 
    batch_size: int
        number of data to load in a particular batch
    shuffle: boolean,optional;
        should shuffle the dataset (default is False)
    num_workers: int,optional
        numbers of workers to run (default is 1)  
    """

    pad_idx = dataset.vocab.stoi["<PAD>"]
    collate_fn = CapsCollate(pad_idx=pad_idx, batch_first=True)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return data_loader


# helper function to save the model
def save_model(model, config):
    model_state = {
        'embed_size': config.embed_size,
        'vocab_size': config.vocab_size,
        'attention_dim': config.attention_dim,
        'encoder_dim': config.encoder_dim,
        'decoder_dim': config.decoder_dim,
        'state_dict': model.state_dict()
    }

    torch.save(model_state, 'attention_model_state.pth')


# generate caption
def get_caps_from(model, features_tensors, vocab, device='cuda'):
    # generate the caption
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(device))
        caps, alphas = model.decoder.generate_caption(features, vocab=vocab)

    return caps, alphas


def generate_and_dump_dataset(root_dir, captions_file, transforms, DATA_LOCATION):
    # initialize the dataset class
    dataset = FlickrDataset(
        root_dir=root_dir,
        captions_file=captions_file,
        transform=transforms
    )

    joblib.dump(dataset, DATA_LOCATION+"/processed_dataset.joblib")


def load_ED_model(model_path):
    # Call: model = load_ED_model('attention_model_state.pth')
    checkpoint = torch.load(model_path)

    model = EncoderDecoder(
        embed_size=checkpoint['embed_size'],
        vocab_size=checkpoint['vocab_size'],
        attention_dim=checkpoint['attention_dim'],
        encoder_dim=checkpoint['encoder_dim'],
        decoder_dim=checkpoint['decoder_dim']
    )
    model.load_state_dict(checkpoint['state_dict'])

    return model




