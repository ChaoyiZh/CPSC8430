import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F


class S2VT(nn.Module):
    def __init__(self, vocab_size, feature_dim, hidden_dim, vocab,max_caption_length, num_layers=1, dropout=0.5,teaching_rate = 1):
        super(S2VT, self).__init__()
        self.max_caption_length = max_caption_length
        self.teaching_rate = teaching_rate

        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab = vocab

        self.drop = nn.Dropout(p=dropout)

        self.linear1 = nn.Linear(in_features=feature_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

        self.lstm1 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                             batch_first=True, dropout=dropout)

        self.lstm2 = nn.LSTM(input_size=hidden_dim*2, hidden_size=hidden_dim, batch_first=True, dropout=dropout)

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)

    def forward(self, video_feats, captions=None):

        self.teaching_rate = max(0, self.teaching_rate - 0.005)

        # Feature encoding
        video_feats = self.drop(video_feats)
        feature_padding = torch.zeros([video_feats.size(0),self.max_caption_length-1, video_feats.size(2)]).to(video_feats.device)
        video_feats_padding = torch.cat([video_feats, feature_padding], dim=1)
        video_feats_padding = self.linear1(video_feats_padding)

        # LSTM 1 - Encoding the video features
        # batch, feature_length+caption_length, 500
        lstm1_out, _ = self.lstm1(video_feats_padding)
        if random.random() < self.teaching_rate and captions is not None:
            # Prepare the inputs for the LSTM 2
            captions_embedded = self.embedding(captions)[:, 0:self.max_caption_length-1, :]
            caption_padding = torch.zeros([captions.size(0),video_feats.size(1), captions_embedded.size(2)]).to(video_feats.device)
            captions_embedded_padding = torch.cat([caption_padding, captions_embedded], dim=1)
            lstm2_in = torch.cat((lstm1_out, captions_embedded_padding), dim=2)

            # LSTM 2 - Generating the caption
            lstm2_out, _ = self.lstm2(lstm2_in)

            # Output layer
            outputs = self.drop(lstm2_out)
            outputs =self.linear2(outputs)
            outputs = outputs[:,-(self.max_caption_length-1):, :]
            return outputs
        #
        else:

            outputs = []

            caption_padding = torch.zeros([video_feats.shape[0], video_feats.size(1), self.hidden_dim]).to(video_feats.device)
            lstm2_in = torch.cat((lstm1_out[:, :video_feats.size(1), :], caption_padding), dim=2)
            lstm2_out, hidden_state = self.lstm2(lstm2_in)

            bos_id = torch.ones(video_feats.shape[0], dtype=torch.long).cuda()
            bos_embed = self.embedding(bos_id)
            lstm2_in = torch.cat((bos_embed, lstm1_out[:, video_feats.size(1), :]), 1).view(-1, 1, 2 * self.hidden_dim)

            lstm2_out, h_state = self.lstm2(lstm2_in, hidden_state)
            lstm2_linear_out = self.linear2(lstm2_out)
            outputs.append(lstm2_linear_out)

            for i in range(self.max_caption_length - 2):
                # decoder_input = self.embedding(x_cap[:, i])
                lstm2_in = torch.cat((lstm1_out[:, video_feats.size(1) + i + 1, :], lstm2_out.view(-1, self.hidden_dim)),
                                           1).view(-1, 1, 2 * self.hidden_dim)
                lstm2_out, h_state = self.lstm2(lstm2_in, h_state)
                # decoder_ouput = decoder_ouput.contiguous().view(-1, self.embed_dim)
                lstm2_out = self.drop(lstm2_out)
                lstm2_linear_out = self.linear2(lstm2_out)
                outputs.append(lstm2_linear_out)
            outputs = torch.cat((outputs), dim=1)
            return outputs


