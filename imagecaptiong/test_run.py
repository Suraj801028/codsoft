import os
import sys
import torch

print('Python', sys.version.replace('\n', ' '))

def try_import(name):
    try:
        m = __import__(name)
        print(f'Imported {name} OK')
        return True
    except Exception as e:
        print(f'Failed to import {name}:', e)
        return False

for mod in ('torch', 'torchvision', 'PIL', 'nltk'):
    try_import(mod)

from dataset import CaptionDataset
from model import DecoderRNN

base = os.path.dirname(__file__)
test_dir = os.path.join(base, 'test_data')
os.makedirs(test_dir, exist_ok=True)

# create a tiny image
from PIL import Image
img_path = os.path.join(test_dir, 'test.jpg')
img = Image.new('RGB', (224, 224), color=(73, 109, 137))
img.save(img_path)

captions_path = os.path.join(test_dir, 'captions.txt')
with open(captions_path, 'w', encoding='utf-8') as f:
    f.write('test.jpg|A test caption for the tiny image.\n')

print('Created test image and captions at', test_dir)

ds = CaptionDataset(test_dir, captions_path)
print('Vocab size:', len(ds.vocab.stoi))
print('Dataset length:', len(ds))

img_tensor, cap_tensor = ds[0]
print('Image tensor shape:', img_tensor.shape)
print('Caption tensor:', cap_tensor)

# test decoder sampling with random features
embed_size = 256
hidden_size = 512
vocab_size = len(ds.vocab.stoi)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
features = torch.randn(1, embed_size)
sampled = decoder.sample(features, ds.vocab, max_len=10)
words = [ds.vocab.itos.get(i, '<unk>') for i in sampled]
print('Sampled ids:', sampled)
print('Sampled words:', words)

print('Test run completed successfully')
