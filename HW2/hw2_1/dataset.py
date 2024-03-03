import torch
from torch.utils.data import Dataset
import numpy as np
import json
import re
from collections import Counter
import os


class VideoFeaturesDataset(Dataset):
    def __init__(self, features_dir, labels_file, max_caption_length= None, vocab=None):
        self.features_dir = features_dir


        self.features_and_captions = []
        if labels_file !=None:
            self.labels = json.load(open(labels_file, 'r'))
        else:
            self.labels = []
            for filename in os.listdir(os.path.join(self.features_dir, "feat")):
                self.labels.append({"caption":[""], "id": os.path.splitext(filename)[0]})


        for item in self.labels:
            video_id = item['id']
            features_path = os.path.join(self.features_dir, "feat", video_id + '.npy')
            features = np.load(features_path)
            for caption in item['caption']:
                self.features_and_captions.append((features, caption, video_id))

        if vocab is None:
            self.vocab, self.max_caption_length = self.build_vocab_and_max_length(
                [caption for item in self.labels for caption in item['caption']])
        else:
            self.vocab = vocab
            self.max_caption_length =max_caption_length

    def build_vocab_and_max_length(self, captions_list):

        vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}

        word_counts = Counter(
            word for caption in captions_list for word in re.sub(r'[^\w\s]', '', caption).lower().split())


        filtered_words = [word for word, count in word_counts.items() if count > 3]


        vocab.update({word: i + 4 for i, word in enumerate(filtered_words)})


        max_length = max(len(caption.split()) for caption in captions_list) + 2

        return vocab, max_length

    def tokenize(self, caption):
        tokens = [self.vocab.get("<start>")] + [self.vocab.get(word, self.vocab["<unk>"]) for word in
                                                re.sub(r'[^\w\s]', '', caption).lower().split()] + [
                     self.vocab.get("<end>")]
        return tokens

    def __len__(self):
        return len(self.features_and_captions)

    def __getitem__(self, idx):
        features, caption, video_id = self.features_and_captions[idx]
        caption_tokens = self.tokenize(caption)
        caption_length = len(caption_tokens)+2
        caption_tokens.extend([self.vocab["<pad>"]] * (self.max_caption_length - len(caption_tokens)))
        caption_tokens_tensor = torch.tensor(caption_tokens, dtype=torch.long)

        return torch.tensor(features, dtype=torch.float),caption_tokens_tensor, video_id, caption_length
