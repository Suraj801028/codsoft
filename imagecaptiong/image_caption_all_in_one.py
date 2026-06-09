"""
Complete Image Captioning System - All in One File
Combines dataset, model, training, inference, and Flask UI
"""

import os
import sys
import re
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter
from tqdm import tqdm
from flask import Flask, request, render_template_string, redirect, url_for, send_from_directory


# ============================================================================
# VOCABULARY AND DATASET
# ============================================================================

def tokenize(text):
    return re.findall(r"\w+|[.,!?;]", text.lower())


class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.freq_threshold = freq_threshold
        self.itos = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        self.stoi = {v: k for k, v in self.itos.items()}

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in tokenize(sentence):
                frequencies[word] += 1
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized = tokenize(text)
        return [self.stoi.get(word, self.stoi['<unk>']) for word in tokenized]


class CaptionDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab=None, transform=None):
        self.root_dir = root_dir
        with open(captions_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        self.samples = [l.split('|', 1) for l in lines]

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        captions = [c for _, c in self.samples]
        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.build_vocabulary(captions)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        path = os.path.join(self.root_dir, img_name)
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        numericalized = [self.vocab.stoi['<start>']] + self.vocab.numericalize(caption) + [self.vocab.stoi['<end>']]
        caption_tensor = torch.tensor(numericalized)
        return image, caption_tensor

    @staticmethod
    def collate_fn(batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        captions = [item[1] for item in batch]
        lengths = [len(c) for c in captions]
        padded = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
        return images, padded, lengths

    def save_vocab(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.vocab, f)

    @staticmethod
    def load_vocab(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ============================================================================
# MODEL: ENCODER AND DECODER
# ============================================================================

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


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(data_dir, captions_file, epochs=5, batch_size=16, embed_size=256, hidden_size=512, lr=1e-3, save_path='caption.pth'):
    dataset = CaptionDataset(data_dir, captions_file)
    vocab = dataset.vocab
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=CaptionDataset.collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

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
            outputs = outputs[:, 1:, :].reshape(-1, outputs.size(2))
            targets = captions[:, 1:].reshape(-1)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'vocab': vocab,
        'embed_size': embed_size,
        'hidden_size': hidden_size
    }, save_path)
    print(f'Model saved to {save_path}')


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def load_model(model_path, device=None):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    vocab = checkpoint.get('vocab')
    embed_size = checkpoint.get('embed_size', checkpoint['encoder']['linear.weight'].size(0))
    hidden_size = checkpoint.get('hidden_size', 512)
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab.stoi))
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    return encoder, decoder, vocab


def caption_image(image_path, model_path, max_len=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder, vocab = load_model(model_path, device)
    encoder.to(device).eval()
    decoder.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = encoder(image)
        sampled_ids = decoder.sample(features, vocab, max_len=max_len)

    words = []
    for idx in sampled_ids:
        word = vocab.itos.get(idx, '<unk>')
        if word == '<end>':
            break
        words.append(word)
    return ' '.join(words)


# ============================================================================
# FLASK WEB UI
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_DEFAULT = os.path.join(BASE_DIR, 'models', 'caption.pth')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Image Captioning UI</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 2rem auto; padding: 1rem; }
    .card { border: 1px solid #ccc; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    input[type=file], input[type=text] { margin-top: 0.5rem; }
    button { padding: 0.6rem 1rem; margin-top: 1rem; }
    img { max-width: 100%; height: auto; margin-top: 1rem; }
    .caption { margin-top: 1rem; font-size: 1.1rem; font-weight: bold; color: #0066cc; }
    .warning { color: #b00; margin-top: 1rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Image Caption Generator</h1>
    <p>Upload an image and generate a caption using the trained model.</p>
    {% if warning %}
      <div class="warning">{{ warning }}</div>
    {% endif %}
    <form method="post" enctype="multipart/form-data">
      <label>Choose image file:</label><br />
      <input type="file" name="image" accept="image/*" required><br />
      <label>Model path:</label><br />
      <input type="text" name="model_path" value="{{ model_path }}" style="width:100%;" /><br />
      <button type="submit">Generate caption</button>
    </form>
    {% if image_url %}
      <img src="{{ image_url }}" alt="Uploaded image" />
    {% endif %}
    {% if caption %}
      <div class="caption">Caption: {{ caption }}</div>
    {% endif %}
  </div>
</body>
</html>
'''


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    image_url = None
    warning = None
    model_path = MODEL_DEFAULT

    if request.method == 'POST':
        if 'image' not in request.files:
            warning = 'No image file part in the request.'
        else:
            file = request.files['image']
            model_path = request.form.get('model_path', MODEL_DEFAULT) or MODEL_DEFAULT
            if file.filename == '':
                warning = 'No image selected.'
            elif not allowed_file(file.filename):
                warning = 'Only PNG, JPG, JPEG and GIF files are allowed.'
            else:
                filename = os.path.basename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                image_url = url_for('uploaded_file', filename=filename)
                if not os.path.exists(model_path):
                    warning = f'Model not found at {model_path}. Please train the model first or provide a valid .pth path.'
                else:
                    try:
                        caption = caption_image(save_path, model_path)
                    except Exception as exc:
                        warning = f'Error generating caption: {exc}'

    return render_template_string(TEMPLATE, caption=caption, image_url=image_url, warning=warning, model_path=model_path)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ============================================================================
# CLI MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Image Captioning System - All in One')
    sub = parser.add_subparsers(dest='cmd')

    # Train subcommand
    t = sub.add_parser('train', help='Train the image captioning model')
    t.add_argument('--data-dir', required=True, help='Directory containing images')
    t.add_argument('--captions', default='captions.txt', help='Captions file name')
    t.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    t.add_argument('--save-path', default='models/caption.pth', help='Path to save model')
    t.add_argument('--batch-size', type=int, default=16, help='Batch size')
    t.add_argument('--embed-size', type=int, default=256, help='Embedding size')
    t.add_argument('--hidden-size', type=int, default=512, help='Hidden size')
    t.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    # Infer subcommand
    i = sub.add_parser('infer', help='Generate caption for an image')
    i.add_argument('--image', required=True, help='Path to image')
    i.add_argument('--model', required=True, help='Path to model checkpoint')

    # UI subcommand
    u = sub.add_parser('ui', help='Launch web UI')
    u.add_argument('--port', type=int, default=5000, help='Port number')

    args = parser.parse_args()

    if args.cmd == 'train':
        captions_file = args.captions
        if not os.path.isabs(captions_file) and not os.path.exists(captions_file):
            captions_file = os.path.join(args.data_dir, args.captions)
        if not os.path.exists(captions_file):
            print(f"Error: captions file not found: {captions_file}")
            sys.exit(1)
        train_model(args.data_dir, captions_file, epochs=args.epochs, batch_size=args.batch_size,
                   embed_size=args.embed_size, hidden_size=args.hidden_size, lr=args.lr, save_path=args.save_path)

    elif args.cmd == 'infer':
        print(caption_image(args.image, args.model))

    elif args.cmd == 'ui':
        print(f'Starting web UI on http://localhost:{args.port}')
        app.run(host='0.0.0.0', port=args.port, debug=True)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
