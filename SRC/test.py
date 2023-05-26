import wandb
import torch
from utils.utils import *
from nltk.translate.bleu_score import sentence_bleu


def test(model, test_loader, vocab, device="cuda", save: bool = True):
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

    for images, captions in test_loader:
        pass

    if save:
        print(len(images))
        print(images.shape)
        print(captions.shape)
        # Save the model in the exchangeable ONNX format
        torch.onnx.export(model,  # model being run
                          (images, captions),  # model input (or a tuple for multiple inputs)
                          "model.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=11,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['images', 'captions'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})
        wandb.save("model.onnx")
