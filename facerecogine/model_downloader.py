import os
import urllib.request
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# URLs and local filenames for the face detection and recognition models
MODELS = {
    'deploy.prototxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel': 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
    'face_detection_yunet_2023mar.onnx': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
    'face_recognition_sface_2021dec.onnx': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx'
}

def report_progress(block_num, block_size, total_size):
    """Callback function to display download progress"""
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    percent = min(100, (downloaded / total_size) * 100)
    sys.stdout.write(f"\rDownloading... {percent:.1f}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
    sys.stdout.flush()

def download_all_models():
    """Download all required models if they don't already exist"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Checking face models in {MODELS_DIR}...")
    
    for filename, url in MODELS.items():
        dest_path = os.path.join(MODELS_DIR, filename)
        
        # Verify file exists and is not empty (corrupt download)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            print(f"[OK] Found: {filename} ({os.path.getsize(dest_path)/(1024*1024):.2f} MB)")
            continue
            
        print(f"\nDownloading {filename} from {url}...")
        try:
            # Add custom user agent to avoid bot blockers
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
            urllib.request.install_opener(opener)
            
            urllib.request.urlretrieve(url, dest_path, report_progress)
            print(f"\n[OK] Saved {filename} to {dest_path}")
        except Exception as e:
            print(f"\n[ERROR] Error downloading {filename}: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise e
            
    print("\nAll model checks completed successfully!")

if __name__ == '__main__':
    try:
        download_all_models()
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
