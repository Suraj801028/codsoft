import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
            features = features.view(features.size(0), -1)
        features = self.linear(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        # prepend image features as first input
        features = features.unsqueeze(1)
        embeddings = torch.cat((features, embeddings), dim=1)
        h0 = self.init_h(features.squeeze(1)).unsqueeze(0)
        c0 = self.init_c(features.squeeze(1)).unsqueeze(0)
        outputs, _ = self.lstm(embeddings, (h0, c0))
        outputs = self.linear(outputs)
        return outputs

    def sample(self, features, vocab, max_len=20):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        h = self.init_h(features).unsqueeze(0)
        c = self.init_c(features).unsqueeze(0)
        for _ in range(max_len):
            outputs, (h, c) = self.lstm(inputs, (h, c))
            outputs = self.linear(outputs.squeeze(1))
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())
            if predicted.item() == vocab.stoi.get('<end>'):
                break
            inputs = self.embed(predicted).unsqueeze(1)
        return sampled_ids
