import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class SiameseEmbedder:
    def __init__(self, use_gpu=True):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"Initializing PyTorch Siamese Embedder on device: {self.device}")
        
        # Load pre-trained ResNet18 model as the backbone feature extractor
        # Use newer torchvision API if available, fallback for older versions
        try:
            from torchvision.models import ResNet18_Weights
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            self.model = models.resnet18(pretrained=True)
            
        # Remove the classification head (fc layer) to get the feature representation
        # ResNet18 outputs a 512-dimensional feature vector after global average pooling
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Define image normalization matching ImageNet training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def get_embedding(self, face_bgr):
        """
        Extract a 512-dimensional embedding vector from a cropped face image (BGR numpy array).
        """
        try:
            # Convert BGR (OpenCV format) to RGB
            face_rgb = face_bgr[:, :, ::-1]
            pil_img = Image.fromarray(face_rgb)
            
            # Preprocess the image and add batch dimension
            tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # Disable gradient calculation for inference
            with torch.no_grad():
                features = self.feature_extractor(tensor_img)
                # Reshape from [1, 512, 1, 1] to [512]
                embedding = features.squeeze().cpu().numpy()
                
            # Normalize embedding to unit length (L2 normalization)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
        except Exception as e:
            print(f"Error extracting embedding in PyTorch: {e}")
            return None

    def compute_similarity(self, emb1, emb2):
        """
        Compute Cosine Similarity between two face embeddings.
        Since they are L2-normalized, cosine similarity is just the dot product.
        Returns a float score in range [-1.0, 1.0].
        """
        if emb1 is None or emb2 is None:
            return 0.0
        return float(np.dot(emb1, emb2))
