Face Recognition Pro
Overview
A desktop application built with Tkinter and OpenCV that provides real‑time face detection, registration, and recognition using multiple deep‑learning models.

Key Features
Multi‑model detection: Haar Cascade, SSD DNN, and YuNet detectors.
Recognition engines: OpenCV SFace and a PyTorch Siamese embedder.
Database management: Automatic loading/saving of user face embeddings (pickle).
Image & video processing:
Capture from webcam or load from file.
Annotate frames with bounding boxes, landmarks, and person names.
Save annotated images and export processed videos.
User management:
Register new users with multiple samples.
Delete users and clean stored images.
Responsive UI:
Dark‑mode theme with modern colour palette.
Progress dialogs for long‑running tasks.
Model handling:
Automatic download of required models if missing.
Lazy loading of models on first use.
How to Run
bash

python ai_face_app.py
Make sure the models/ folder contains the required model files (the app will download them if absent).

Dependencies
Python 3.13
OpenCV, Pillow, Tkinter, NumPy, PyTorch, tqdm, etc.
License
MIT License – feel free to modify and distribute.
