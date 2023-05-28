import wandb
import torch
from utils.utils import *
from nltk.translate.bleu_score import sentence_bleu


def test(model, test_loader, criterion, vocab, config, device="cuda", verbatim=True):
    # Run the model on some test examples
    acc_arr_batch = []
    loss_arr_batch = []
    total = 0
    with torch.no_grad():
        for images, captions in test_loader:
            images, captions = images.to(device), captions.to(device)

            # Calculating loss
            outputs, attentions = model(images, captions)
            targets = captions[:, 1:]
            loss = criterion(outputs.view(-1, config.vocab_size), targets.reshape(-1))

            # Calculating accuracy
            images = images[0].detach().clone()
            predicted, _ = get_caps_from(model, images.unsqueeze(0), vocab=vocab, device=device)

            caps = [vocab.get_caption(cap.tolist()) for cap in captions]

            acc_score = sentence_bleu(caps, predicted)

            # Appending metrics
            acc_arr_batch.append(acc_score)
            loss_arr_batch.append(loss.tolist())

            total += 1

            # Report metrics every 1th batch
            if ((total + 1) % 25) == 0 and verbatim:
                print("Batch:", total, "\nAcc_score = ", acc_score)
                print("Loss:", loss.tolist())


        print(f"Mean BLEU score of the model on the {total*5} " +
              f"test images: {sum(acc_arr_batch)/len(acc_arr_batch)}%")
        
        wandb.log({"test_mean_bleu": sum(acc_arr_batch)/len(acc_arr_batch)})

    return acc_arr_batch, loss_arr_batch
