import torch
from PIL import Image
from torchvision import transforms
from dataset import CaptionDataset
from model import EncoderCNN, DecoderRNN


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    print(caption_image(args.image, args.model))
