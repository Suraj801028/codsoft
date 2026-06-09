# Image Captioning (ResNet + RNN)

Simple image captioning example using a pretrained ResNet encoder and an LSTM decoder.

Dataset format
- Place images under `images/`.
- Create `captions.txt` with lines: `image.jpg|a caption describing the image.`

Quick start

Install:
```bash
pip install -r requirements.txt
```

Train:
```bash
cd c:\Users\acer\OneDrive\Desktop\codsoft\imagecaptiong
python image_captiong.py train --data-dir data --captions captions.txt --epochs 10 --save-path models/caption.pth
```

Infer:
```bash
cd c:\Users\acer\OneDrive\Desktop\codsoft\imagecaptiong
python image_captiong.py infer --image path/to/img.jpg --model models/caption.pth
```

UI:
```bash
cd c:\Users\acer\OneDrive\Desktop\codsoft\imagecaptiong
python image_captiong.py ui --port 5000
```

Files
- `dataset.py` - dataset and vocabulary utilities
- `model.py` - encoder and decoder implementations
- `train.py` - training loop
- `inference.py` - run caption generation on an image
- `image_captiong.py` - CLI entrypoint
