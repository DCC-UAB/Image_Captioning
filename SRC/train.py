from tqdm.auto import tqdm
import wandb
from utils.utils import *
from test import *


def train(model, data_loader, criterion, optimizer, config, epoch, verbatim = True): # 25
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    example_ct = 0  # number of examples seen
    batch_ct = 0

    loss_arr_batch = []  # Losses of the batches

    for idx, (image, captions) in enumerate(iter(data_loader)):

        loss = train_batch(image.to(torch.float32), captions, model, config.vocab_size, optimizer, criterion, device=config.device)
        example_ct += len(image)
        batch_ct += 1

        loss_arr_batch.append(loss.tolist())

        # Report metrics every 1th batch
        if ((batch_ct + 1) % 1) == 0 and verbatim:
            train_log(loss, example_ct, epoch)

    return loss_arr_batch


def train_batch(image, captions, model, vocab_size, optimizer, criterion, device='cuda'):

    image, captions = image.to(device), captions.to(device)

    # Zero the gradients.
    optimizer.zero_grad()

    # Feed forward
    outputs, attentions = model(image, captions)

    # Calculate the batch loss.
    targets = captions[:, 1:]

    loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))

    # Backward pass.
    loss.backward()

    # Update the parameters in the optimizer.
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
