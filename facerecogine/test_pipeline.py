import os
import cv2
import numpy as np
import torch
import torchvision

# Import project modules
from model_downloader import download_all_models
from pytorch_siamese import SiameseEmbedder
from face_pipeline import FacePipeline

def run_tests():
    print("=" * 60)
    print("RUNNING AUTOMATED FACE PIPELINE TESTS")
    print("=" * 60)
    
    # Test 1: Download Model files
    print("\n[TEST 1] Downloading deep learning model weights...")
    try:
        download_all_models()
        print("[OK] Model download check successful.")
    except Exception as e:
        print(f"[ERROR] Model download failed: {e}")
        return False

    # Test 2: PyTorch Siamese Embedder loading & feature extraction
    print("\n[TEST 2] Testing PyTorch Siamese Face Embedder...")
    try:
        embedder = SiameseEmbedder(use_gpu=False)
        print("[OK] SiameseEmbedder loaded successfully.")
        
        # Test feature extraction on mock face crop
        mock_face = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
        emb = embedder.get_embedding(mock_face)
        
        if emb is not None:
            print(f"[OK] Extracted mock embedding size: {emb.shape}")
            # Verify shape
            assert emb.shape == (512,), f"Expected 512-D embedding, got {emb.shape}"
            # Verify L2 normalization
            norm = np.linalg.norm(emb)
            print(f"[OK] Embedding L2 Norm: {norm:.4f}")
            assert np.allclose(norm, 1.0, atol=1e-5), "Embedding should be L2 normalized to unit length"
            
            # Test similarity metric
            sim = embedder.compute_similarity(emb, emb)
            print(f"[OK] Cosine similarity of embedding with itself: {sim:.4f}")
            assert np.allclose(sim, 1.0, atol=1e-5), "Cosine similarity of identical vectors should be 1.0"
            print("[OK] PyTorch feature extraction metrics verified successfully.")
        else:
            print("[ERROR] Failed to extract embedding.")
            return False
    except Exception as e:
        print(f"[ERROR] PyTorch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Face Pipeline loading
    print("\n[TEST 3] Loading FacePipeline wrapper...")
    try:
        pipeline = FacePipeline()
        print("[OK] FacePipeline initialized successfully.")
        
        # Check loaded models
        print(f"  - Haar Cascade: {'Loaded' if pipeline.detector_cascade else 'Not Loaded'}")
        print(f"  - ResNet-10 SSD: {'Loaded' if pipeline.detector_ssd else 'Not Loaded'}")
        print(f"  - YuNet Detector: {'Loaded' if pipeline.detector_yunet else 'Not Loaded'}")
        print(f"  - SFace Recognizer: {'Loaded' if pipeline.recognizer_sface else 'Not Loaded'}")
        print(f"  - PyTorch Siamese: {'Loaded' if pipeline.recognizer_pytorch else 'Not Loaded'}")
        
        # Run detection on mock frame (should return empty list since no faces are in random noise)
        mock_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        for detector_name in ["SSD DNN", "YuNet", "Haar Cascade"]:
            detections = pipeline.detect_faces(mock_frame, model_name=detector_name, conf_threshold=0.5)
            print(f"  - {detector_name} detection mock run: {len(detections)} faces found.")
            assert isinstance(detections, list), f"Expected list of detections, got {type(detections)}"
            
        print("[OK] FacePipeline model verification complete.")
    except Exception as e:
        print(f"[ERROR] FacePipeline test failed: {e}")
        return False
        
    print("\n" + "=" * 60)
    print("ALL AUTOMATED TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = run_tests()
    import sys
    sys.exit(0 if success else 1)
