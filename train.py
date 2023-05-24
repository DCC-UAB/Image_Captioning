from tqdm.auto import tqdm
import wandb


def train(model, data_loader, criterion, optimizer, vocab_size, num_epochs=25):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0

    for epoch in tqdm(range(1, num_epochs + 1)):
        for idx, (image, captions) in enumerate(iter(data_loader)):

            loss = train_batch(image, captions, model, vocab_size, optimizer, criterion)
            example_ct += len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

        # save the latest model
        save_model(model, epoch)


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
