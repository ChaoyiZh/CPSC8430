import os
from train import *
from model_seq2seq_model import *
from dataset import *
import sys
import pickle
from utils import detokenize
from argparse import ArgumentParser

parser = ArgumentParser(description="Training options: ")

parser.add_argument('--features_dir', type=str,
                    default='/home/chaoyi/workspace/course/CPSC8430/HW2/hw2_1/data/MLDS_hw2_1_data/testing_data',
                    required=True)

parser.add_argument('--output_file', type=str,
                    default='testset_output.txt',
                    required=True)


args = parser.parse_args()
vocab_path = './vocab.pkl'
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)


test_set = VideoFeaturesDataset(args.features_dir, labels_file=None,vocab=vocab,max_caption_length=42)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False,drop_last=True)

model = S2VT(vocab_size =len(vocab), feature_dim=test_set[0][0].shape[1], vocab=vocab,max_caption_length = test_set.max_caption_length,hidden_dim=500,teaching_rate=0.8)
model.load_state_dict(torch.load('model_weights.pth'))
model.to(device)
model.eval()
results = []
with torch.no_grad():
    for batch_idx, (video_feats, captions,video_ids,caption_length) in enumerate(test_loader):
        video_feats, captions = video_feats.to(device), captions.to(device)


        outputs = model(video_feats)

        print("The prediction of the first data in minibatch: ")
        outputs = torch.argmax(outputs, dim=-1)
        result = detokenize(outputs, vocab, video_ids)
        results.append(result)

with open(args.output_file, 'w') as f:
    f.writelines(results)



