import wandb
import torch
from utils.utils import *
from nltk.translate.bleu_score import sentence_bleu


def test(model, test_loader, vocab, device="cuda"):
    # Run the model on some test examples
    with torch.no_grad():
        acc_score, total = 0, 0
        for images, captions in test_loader:
            images, captions = images.to(device), captions.to(device)
            images = images[0].detach().clone()
            predicted, _ = get_caps_from(model, images.unsqueeze(0), vocab=vocab, device=device)
            acc_score += sentence_bleu(captions, predicted)
            total += 1

            # Report metrics every 1th batch
            if ((total + 1) % 1) == 0:
                print("Batch:", total, "\nAcc_score = ", 100*acc_score/total)

        print(f"Mean BLEU score of the model on the {total} " +
              f"test images: {acc_score / total * 100}%")
        
        wandb.log({"test_mean_bleu": acc_score / total * 100})

