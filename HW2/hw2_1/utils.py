import torch
from itertools import takewhile

from torch import nn


def eval(epoch, args, model,test_loader, device, vocab):
    print("Evaluating the model")
    model.eval()
    # criterion = nn.CrossEntropyLoss()
    # total_loss = 0
    with torch.no_grad():
        for batch_idx, (video_feats, captions,video_ids,caption_length) in enumerate(test_loader):
            video_feats, captions = video_feats.to(device), captions.to(device)


            outputs = model(video_feats)

            if batch_idx%200== 0:
                print("The label of the first data in minibatch: ")
                detokenize(captions[:, 1:], vocab, video_ids)
                print("The prediction of the first data in minibatch: ")
                outputs = torch.argmax(outputs, dim=-1)
                detokenize(outputs, vocab, video_ids)




def detokenize(captions, vocab, video_ids):
    endsyntax = ["<end>", "<pad>"]

    id_word_dict = {value: key for key, value in vocab.items()}


    # batchsize, max_length
    captions_tokens = captions
    captions_token = captions_tokens[0]
    captions = [id_word_dict[int(token.cpu().numpy())] for token in captions_token]
    filtered_captions = []
    for word in captions:
        if word == endsyntax[0]:
            break
        if word not in endsyntax:
            filtered_captions.append(word)

    print(video_ids[0]+"," + " ".join(filtered_captions))
    return str(video_ids[0]+"," + " ".join(filtered_captions)+ "\n")

