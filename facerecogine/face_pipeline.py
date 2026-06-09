import os
import cv2
import numpy as np
import pickle
from model_downloader import MODELS_DIR
from pytorch_siamese import SiameseEmbedder

class FacePipeline:
    def __init__(self, known_faces_dir=None, db_file=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if known_faces_dir is None:
            known_faces_dir = os.path.join(base_dir, 'known_faces')
        if db_file is None:
            db_file = os.path.join(base_dir, 'face_db.pkl')
            
        self.known_faces_dir = known_faces_dir
        self.db_file = db_file
        self.database = {}  # Format: {username: {'sface_embeddings': [], 'pytorch_embeddings': []}}
        
        # Paths to models
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.ssd_proto = os.path.join(MODELS_DIR, 'deploy.prototxt')
        self.ssd_model = os.path.join(MODELS_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
        self.yunet_model = os.path.join(MODELS_DIR, 'face_detection_yunet_2023mar.onnx')
        self.sface_model = os.path.join(MODELS_DIR, 'face_recognition_sface_2021dec.onnx')
        
        # Loaded model references
        self.detector_cascade = None
        self.detector_ssd = None
        self.detector_yunet = None
        self.recognizer_sface = None
        self.recognizer_pytorch = None
        
        # Load models and database
        self.load_models()
        self.load_database()
        
    def load_models(self):
        """Lazy load models as needed or load them initially if files exist"""
        # 1. Load Haar Cascades
        if os.path.exists(self.cascade_path):
            self.detector_cascade = cv2.CascadeClassifier(self.cascade_path)
            
        # 2. Load ResNet-10 SSD
        if os.path.exists(self.ssd_proto) and os.path.exists(self.ssd_model):
            try:
                self.detector_ssd = cv2.dnn.readNetFromCaffe(self.ssd_proto, self.ssd_model)
            except Exception as e:
                print(f"Failed to load ResNet-10 SSD: {e}")
                
        # 3. Load YuNet (Detector size is set dynamically on first frame)
        if os.path.exists(self.yunet_model):
            try:
                # We instantiate with a dummy input size, will update on detect
                self.detector_yunet = cv2.FaceDetectorYN.create(self.yunet_model, "", (320, 320))
            except Exception as e:
                print(f"Failed to load YuNet: {e}")
                
        # 4. Load SFace
        if os.path.exists(self.sface_model):
            try:
                self.recognizer_sface = cv2.FaceRecognizerSF.create(self.sface_model, "")
            except Exception as e:
                print(f"Failed to load SFace: {e}")
                
        # 5. Load PyTorch Siamese Embedder
        try:
            self.recognizer_pytorch = SiameseEmbedder()
        except Exception as e:
            print(f"Failed to load PyTorch Siamese Embedder: {e}")
            
    def load_database(self):
        """Load registered users database from pickle file or create it by processing images"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'rb') as f:
                    self.database = pickle.load(f)
                print(f"Loaded database: {len(self.database)} users registered.")
                return
            except Exception as e:
                print(f"Error loading {self.db_file}: {e}. Rebuilding database...")
                
        self.rebuild_database()
        
    def save_database(self):
        """Save database to pickle file"""
        try:
            with open(self.db_file, 'wb') as f:
                pickle.dump(self.database, f)
            print(f"Saved database to {self.db_file}")
        except Exception as e:
            print(f"Failed to save database: {e}")
            
    def rebuild_database(self):
        """Scan known_faces directory and compute encodings for all users"""
        self.database = {}
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir, exist_ok=True)
            return
            
        print("Rebuilding face database. Extracting features...")
        for name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, name)
            if not os.path.isdir(person_dir):
                continue
                
            self.database[name] = {'sface_embeddings': [], 'pytorch_embeddings': []}
            
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                if not os.path.isfile(img_path):
                    continue
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Detect the face in this image using YuNet (or fallbacks)
                faces = self.detect_faces_internal(img)
                if len(faces) == 0:
                    print(f"Warning: No face detected in registration image: {img_path}")
                    continue
                    
                # Take the largest face in case of multiple
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                face_data = faces[0]
                
                # Get embeddings
                sface_emb = self.extract_sface_embedding(img, face_data)
                pytorch_emb = self.extract_pytorch_embedding(img, face_data[0:4])
                
                if sface_emb is not None:
                    self.database[name]['sface_embeddings'].append(sface_emb)
                if pytorch_emb is not None:
                    self.database[name]['pytorch_embeddings'].append(pytorch_emb)
                    
            print(f"Processed registered user: {name} ({len(self.database[name]['pytorch_embeddings'])} samples)")
            
        self.save_database()
        
    def register_user(self, name, frame):
        """
        Register a single face crop from a frame for a user.
        Appends the face template to the user's database entry and saves the image to disk.
        """
        person_dir = os.path.join(self.known_faces_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        
        faces = self.detect_faces_internal(frame)
        if len(faces) == 0:
            return False, "No face detected in the image."
            
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        face_data = faces[0]
        
        # Save image file to disk
        timestamp = cv2.getTickCount()
        img_path = os.path.join(person_dir, f"{name}_{timestamp}.jpg")
        cv2.imwrite(img_path, frame)
        
        # Initialize user in database if not exists
        if name not in self.database:
            self.database[name] = {'sface_embeddings': [], 'pytorch_embeddings': []}
            
        # Compute embeddings
        sface_emb = self.extract_sface_embedding(frame, face_data)
        pytorch_emb = self.extract_pytorch_embedding(frame, face_data[0:4])
        
        if sface_emb is not None:
            self.database[name]['sface_embeddings'].append(sface_emb)
        if pytorch_emb is not None:
            self.database[name]['pytorch_embeddings'].append(pytorch_emb)
            
        self.save_database()
        return True, f"Successfully registered {name}."

    def delete_user(self, name):
        """Delete a user from the database and remove their registered images folder"""
        if name in self.database:
            del self.database[name]
            self.save_database()
            
        person_dir = os.path.join(self.known_faces_dir, name)
        if os.path.exists(person_dir):
            import stat
            try:
                # Walk bottom-up to change permissions and delete files/directories
                for root, dirs, files in os.walk(person_dir, topdown=False):
                    for f in files:
                        filepath = os.path.join(root, f)
                        os.chmod(filepath, stat.S_IWRITE)
                        os.remove(filepath)
                    for d in dirs:
                        dirpath = os.path.join(root, d)
                        os.chmod(dirpath, stat.S_IWRITE)
                        os.rmdir(dirpath)
                os.chmod(person_dir, stat.S_IWRITE)
                os.rmdir(person_dir)
            except Exception as e:
                print(f"Error removing directory {person_dir}: {e}")
        return True

    def detect_faces_internal(self, frame):
        """
        Internal face detector using YuNet (fallback to SSD, then Cascade) to locate faces
        and return standard face arrays for feature extraction.
        Format of returned list items: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, conf]
        """
        h, w = frame.shape[:2]
        
        # Try YuNet first (provides accurate landmarks for SFace alignment)
        if self.detector_yunet is not None:
            try:
                self.detector_yunet.setInputSize((w, h))
                retval, faces = self.detector_yunet.detect(frame)
                if retval and faces is not None:
                    # Convert to list of floats
                    return [f for f in faces]
            except Exception as e:
                print(f"YuNet internal detection failed: {e}")
                
        # Try SSD Caffe next
        if self.detector_ssd is not None:
            try:
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                self.detector_ssd.setInput(blob)
                detections = self.detector_ssd.forward()
                
                faces = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x1, y1, x2, y2) = box.astype("int")
                        x, y, width, height = x1, y1, x2 - x1, y2 - y1
                        
                        # Mock YuNet landmark array of size 15
                        f = np.zeros(15, dtype=np.float32)
                        f[0:4] = [x, y, width, height]
                        # Mock landmarks
                        f[4] = x + width * 0.3
                        f[5] = y + height * 0.4
                        f[6] = x + width * 0.7
                        f[7] = y + height * 0.4
                        f[8] = x + width * 0.5
                        f[9] = y + height * 0.6
                        f[10] = x + width * 0.35
                        f[11] = y + height * 0.8
                        f[12] = x + width * 0.65
                        f[13] = y + height * 0.8
                        f[14] = confidence
                        faces.append(f)
                return faces
            except Exception as e:
                print(f"SSD internal detection failed: {e}")
                
        # Try Haar Cascade last
        if self.detector_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self.detector_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            faces = []
            for (x, y, width, height) in detections:
                f = np.zeros(15, dtype=np.float32)
                f[0:4] = [x, y, width, height]
                # Mock landmarks
                f[4] = x + width * 0.3
                f[5] = y + height * 0.4
                f[6] = x + width * 0.7
                f[7] = y + height * 0.4
                f[8] = x + width * 0.5
                f[9] = y + height * 0.6
                f[10] = x + width * 0.35
                f[11] = y + height * 0.8
                f[12] = x + width * 0.65
                f[13] = y + height * 0.8
                f[14] = 1.0
                faces.append(f)
            return faces
            
        return []
        
    def detect_faces(self, frame, model_name='SSD DNN', conf_threshold=0.5):
        """
        Unified face detection interface.
        Returns a list of dicts: [{'box': (x,y,w,h), 'landmarks': [(x,y), ...], 'confidence': float}]
        """
        h, w = frame.shape[:2]
        results = []
        
        if model_name == 'Haar Cascade':
            if self.detector_cascade is None:
                return []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self.detector_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            for (x, y, width, height) in detections:
                results.append({
                    'box': (int(x), int(y), int(width), int(height)),
                    'landmarks': [],
                    'confidence': 1.0
                })
                
        elif model_name == 'SSD DNN':
            if self.detector_ssd is None:
                return []
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.detector_ssd.setInput(blob)
            detections = self.detector_ssd.forward()
            
            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence >= conf_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    # Clip coordinates inside frame
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w - 1, x2)
                    y2 = min(h - 1, y2)
                    
                    results.append({
                        'box': (x1, y1, x2 - x1, y2 - y1),
                        'landmarks': [],
                        'confidence': confidence
                    })
                    
        elif model_name == 'YuNet':
            if self.detector_yunet is None:
                return []
            self.detector_yunet.setInputSize((w, h))
            retval, faces = self.detector_yunet.detect(frame)
            if retval and faces is not None:
                for f in faces:
                    confidence = float(f[14])
                    if confidence >= conf_threshold:
                        x, y, width, height = map(int, f[0:4])
                        # Landmarks
                        landmarks = []
                        for j in range(5):
                            lx = int(f[4 + 2*j])
                            ly = int(f[5 + 2*j])
                            landmarks.append((lx, ly))
                        results.append({
                            'box': (x, y, width, height),
                            'landmarks': landmarks,
                            'confidence': confidence,
                            'raw_face_data': f  # Used for SFace alignment
                        })
                        
        return results

    def extract_sface_embedding(self, frame, face_data):
        """Extract face embedding using SFace model"""
        if self.recognizer_sface is None:
            return None
        try:
            # Align the face crop using SFace
            # SFace alignCrop accepts the raw face_data array returned by YuNet
            aligned_face = self.recognizer_sface.alignCrop(frame, face_data)
            embedding = self.recognizer_sface.feature(aligned_face)
            return embedding
        except Exception as e:
            print(f"SFace feature extraction failed: {e}")
            return None

    def extract_pytorch_embedding(self, frame, box):
        """Extract face embedding using PyTorch Siamese network"""
        if self.recognizer_pytorch is None:
            return None
        try:
            x, y, w, h = map(int, box)
            # Clip bounding box coordinates to frame size
            fh, fw = frame.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(fw - 1, x + w)
            y2 = min(fh - 1, y + h)
            
            # Crop the face ROI
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                return None
                
            return self.recognizer_pytorch.get_embedding(face_roi)
        except Exception as e:
            print(f"PyTorch feature extraction failed: {e}")
            return None

    def process_and_recognize(self, frame, detector_name='SSD DNN', recognizer_name='SFace DNN', conf_threshold=0.5, similarity_threshold=0.4):
        """
        Detects faces in a frame, extracts features, and matches against database.
        Returns a list of face results: [{'box': (x,y,w,h), 'landmarks': [], 'name': str, 'similarity': float, 'confidence': float}]
        """
        # Run detection
        detections = self.detect_faces(frame, model_name=detector_name, conf_threshold=conf_threshold)
        
        results = []
        for det in detections:
            x, y, w, h = det['box']
            confidence = det['confidence']
            landmarks = det['landmarks']
            
            best_name = "Unknown"
            best_similarity = 0.0
            
            # Skip recognition if database is empty
            if self.database:
                # 1. Perform OpenCV SFace recognition
                if recognizer_name == 'SFace DNN' and self.recognizer_sface is not None:
                    # Construct/get face data with landmarks
                    if 'raw_face_data' in det:
                        face_data = det['raw_face_data']
                    else:
                        # Construct a mock face_data for alignCrop
                        face_data = np.zeros(15, dtype=np.float32)
                        face_data[0:4] = [x, y, w, h]
                        # Mock landmarks
                        face_data[4] = x + w * 0.3
                        face_data[5] = y + h * 0.4
                        face_data[6] = x + w * 0.7
                        face_data[7] = y + h * 0.4
                        face_data[8] = x + w * 0.5
                        face_data[9] = y + h * 0.6
                        face_data[10] = x + w * 0.35
                        face_data[11] = y + h * 0.8
                        face_data[12] = x + w * 0.65
                        face_data[13] = y + h * 0.8
                        face_data[14] = confidence
                        
                    emb = self.extract_sface_embedding(frame, face_data)
                    if emb is not None:
                        # Find best match in DB
                        for name, templates in self.database.items():
                            sface_embs = templates.get('sface_embeddings', [])
                            for template_emb in sface_embs:
                                # Match using cosine similarity
                                score = float(self.recognizer_sface.match(emb, template_emb, cv2.FaceRecognizerSF_FR_COSINE))
                                if score > best_similarity:
                                    best_similarity = score
                                    best_name = name
                                    
                # 2. Perform PyTorch Siamese recognition
                elif recognizer_name == 'PyTorch Siamese' and self.recognizer_pytorch is not None:
                    emb = self.extract_pytorch_embedding(frame, (x, y, w, h))
                    if emb is not None:
                        for name, templates in self.database.items():
                            pytorch_embs = templates.get('pytorch_embeddings', [])
                            for template_emb in pytorch_embs:
                                score = self.recognizer_pytorch.compute_similarity(emb, template_emb)
                                if score > best_similarity:
                                    best_similarity = score
                                    best_name = name
                                    
            # Apply threshold to determine recognition status
            if best_similarity < similarity_threshold:
                final_name = "Unknown"
            else:
                final_name = best_name
                
            results.append({
                'box': (x, y, w, h),
                'landmarks': landmarks,
                'name': final_name,
                'similarity': best_similarity,
                'confidence': confidence
            })
            
        return results
