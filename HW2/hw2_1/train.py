import torch

from argparse import ArgumentParser
from dataset import VideoFeaturesDataset
from torch.utils.data import DataLoader
from model_seq2seq_model import S2VT
from torch.utils.tensorboard import SummaryWriter
from utils import *
import pickle
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    start_time = time.time()

    train_set = VideoFeaturesDataset(args.features_dir, args.labels_file)
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(train_set.vocab, f)

    test_set = VideoFeaturesDataset(args.features_dir.replace("training_data", "testing_data"), args.labels_file.replace("training_label", "testing_label"),vocab=train_set.vocab,max_caption_length=train_set.max_caption_length)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False,drop_last=True)

    model = S2VT(vocab_size =len(train_set.vocab), feature_dim=train_set[0][0].shape[1], vocab= train_set.vocab,max_caption_length = train_set.max_caption_length,hidden_dim=500,teaching_rate=0.8)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    losses = []

    for epoch in range(args.num_epoches):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        for batch_idx, (video_feats, captions, video_ids, caption_length) in enumerate(train_loader):
            # batch* feature_size,  batch*max_length
            video_feats, captions = video_feats.to(device), captions.to(device)




        #
        #     # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(video_feats, captions)
            loss = criterion(outputs.reshape(-1, len(train_set.vocab)), captions[:,1:].contiguous().view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Training Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
                print("The label of the first data in minibatch: ")
                detokenize(captions[:,1:], train_set.vocab, video_ids)
                print("The prediction of the first data in minibatch: ")
                detokenize(torch.argmax(outputs, dim=2), train_set.vocab, video_ids)

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('training loss', avg_loss, epoch)
        print(
            f'Training Epoch [{epoch + 1}/{args.num_epoches}], Loss: {avg_loss:.4f}, Time: {time.time() - epoch_start_time:.2f} seconds')
        losses.append(avg_loss)

        eval(epoch,args, model,
                                    test_loader,
                                    device,
                                    train_set.vocab,)

    torch.save(model.state_dict(), 'model_weights.pth')
    print(f"Training finished in {time.time() - start_time:.2f} seconds")





if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Training options: ")

    parser.add_argument('--features_dir', type=str,
                        default='/home/chaoyi/workspace/course/CPSC8430/HW2/hw2_1/data/MLDS_hw2_1_data/training_data',
                        required=True)

    parser.add_argument('--labels_file', type=str,
                        default='/home/chaoyi/workspace/course/CPSC8430/HW2/hw2_1/data/MLDS_hw2_1_data/training_label.json',
                        required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epoches', type=int, default=200)
    args = parser.parse_args()
    writer = SummaryWriter('runs/experiment_1')
    train(args)

    print("Training finished")


