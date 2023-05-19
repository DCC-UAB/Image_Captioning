from tqdm.auto import tqdm
import wandb
#
#def train(model, loader, criterion, optimizer, config):
#    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
#    wandb.watch(model, criterion, log="all", log_freq=10)
#
#    # Run training and track with wandb
#    total_batches = len(loader) * config.epochs
#    example_ct = 0  # number of examples seen
#    batch_ct = 0
#    for epoch in tqdm(range(config.epochs)):
#        for _, (images, labels) in enumerate(loader):
#
#            loss = train_batch(images, labels, model, optimizer, criterion)
#            example_ct +=  len(images)
#            batch_ct += 1
#
#            # Report metrics every 25th batch
#            if ((batch_ct + 1) % 25) == 0:
#                train_log(loss, example_ct, epoch)
#
#
#def train_batch(images, labels, model, optimizer, criterion, device="cuda"):
#    images, labels = images.to(device), labels.to(device)
#    
#    # Forward pass ➡ 
#    outputs = model(images)
#    loss = criterion(outputs, labels)
#    
#    # Backward pass ⬅
#    optimizer.zero_grad()
#    loss.backward()
#
#    # Step with optimizer
#    optimizer.step()
#
#    return loss
#
#
#def train_log(loss, example_ct, epoch):
#    # Where the magic happens
#    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
#    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")


# Hyperparams
embed_size=300
vocab_size = len(dataset.vocab)
attention_dim=256
encoder_dim=2048
decoder_dim=512
learning_rate = 3e-4


# init model
model = EncoderDecoder(
    embed_size=300,
    vocab_size = len(dataset.vocab),
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
).to(device)

# Training model
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
num_epochs = 25
print_every = 100

for epoch in range(1,num_epochs+1):   
    for idx, (image, captions) in enumerate(iter(data_loader)):
        # Loading new batch
        image,captions = image.to(device),captions.to(device)

        # Zero the gradients.
        optimizer.zero_grad()

        # Feed forward
        outputs, attentions = model(image, captions)

        # Calculate the batch loss.
        targets = captions[:,1:]
        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
        
        # Backward pass.
        loss.backward()

        # Update the parameters in the optimizer.
        optimizer.step()

        # Print state of the training
        if (idx+1)%print_every == 0:
            print("Epoch: {} loss: {:.5f}".format(epoch,loss.item()))
            
            #generate the caption
            model.eval() # Canviem a eval per a generar la caption
            with torch.no_grad():
                dataiter = iter(data_loader)
                img,_ = next(dataiter)
                features = model.encoder(img[0:1].to(device))
                caps,alphas = model.decoder.generate_caption(features,vocab=dataset.vocab)
                caption = ' '.join(caps)
                show_image(img[0],title=caption)
                
            model.train() # Tornem a model Train per a seguir entrenant
        
    #save the latest model
    save_model(model,epoch)