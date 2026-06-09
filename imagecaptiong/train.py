import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CaptionDataset
from model import EncoderCNN, DecoderRNN
from tqdm import tqdm


def train(data_dir, captions_file, epochs=5, batch_size=16, embed_size=256, hidden_size=512, lr=1e-3, save_path='caption.pth'):
    dataset = CaptionDataset(data_dir, captions_file)
    vocab = dataset.vocab
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=CaptionDataset.collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab.stoi)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<pad>'])
    params = list(decoder.parameters()) + list(encoder.linear.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        loop = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, captions, lengths in loop:
            images, captions = images.to(device), captions.to(device)
            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])
            # outputs: (batch, seq_len+1, vocab)
            outputs = outputs[:, 1:, :].reshape(-1, outputs.size(2))
            targets = captions[:, 1:].reshape(-1)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(), 'vocab': vocab, 'embed_size': embed_size, 'hidden_size': hidden_size}, save_path)
    print('Model saved to', save_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--captions', default='captions.txt')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--save-path', default='models/caption.pth')
    args = parser.parse_args()
    train(args.data_dir, os.path.join(args.data_dir, args.captions), epochs=args.epochs, save_path=args.save_path)
