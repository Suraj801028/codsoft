import os
import sys
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import cv2
from PIL import Image, ImageTk
import torch

# Import project modules
from model_downloader import download_all_models, MODELS_DIR
from face_pipeline import FacePipeline
import utils

# Theme Palette (Modern Premium Dark Mode)
BG_COLOR = "#121212"       # Dark charcoal background
CARD_BG = "#1e1e1e"        # Light charcoal for panels/cards
BORDER_COLOR = "#2d2d2d"   # Dark grey borders
TEXT_COLOR = "#ffffff"     # White text
MUTED_TEXT = "#888888"     # Medium grey for secondary labels
ACCENT_COLOR = "#3498db"   # Electric flat blue
ACCENT_HOVER = "#2980b9"   # Darker blue for hovers
SUCCESS_COLOR = "#2ecc71"  # Flat emerald green
DANGER_COLOR = "#e74c3c"   # Flat alizarin red


class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Pro")
        self.root.geometry("1150x780")
        self.root.configure(bg=BG_COLOR)
        
        # Initialize variables
        self.pipeline = None
        self.webcam_active = False
        self.webcam_source = 0
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.worker_thread = None
        self.cancel_video_proc = False
        
        # Load custom styles
        self.setup_styles()
        
        # Create Splash/Download screen first
        self.show_splash_screen()
        
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure overall style options
        self.style.configure('.', background=BG_COLOR, foreground=TEXT_COLOR, font=('Segoe UI', 10))
        
        # Frames
        self.style.configure('TFrame', background=BG_COLOR)
        self.style.configure('Card.TFrame', background=CARD_BG, borderwidth=1, relief='solid')
        self.style.map('Card.TFrame', bordercolor=[('focus', ACCENT_COLOR), ('!focus', BORDER_COLOR)])
        
        # Notebook (Tabs)
        self.style.configure('TNotebook', background=BG_COLOR, borderwidth=0)
        self.style.configure('TNotebook.Tab', background=CARD_BG, foreground=MUTED_TEXT, 
                             padding=[12, 6], font=('Segoe UI', 10, 'bold'))
        self.style.map('TNotebook.Tab', 
                       background=[('selected', ACCENT_COLOR), ('active', BORDER_COLOR)],
                       foreground=[('selected', TEXT_COLOR), ('active', TEXT_COLOR)])
                       
        # Buttons
        self.style.configure('TButton', background=ACCENT_COLOR, foreground=TEXT_COLOR, 
                             borderwidth=0, padding=[12, 6], font=('Segoe UI', 10, 'bold'))
        self.style.map('TButton', background=[('active', ACCENT_HOVER), ('disabled', BORDER_COLOR)])
        
        self.style.configure('Success.TButton', background=SUCCESS_COLOR, foreground=TEXT_COLOR)
        self.style.map('Success.TButton', background=[('active', "#27ae60")])
        
        self.style.configure('Danger.TButton', background=DANGER_COLOR, foreground=TEXT_COLOR)
        self.style.map('Danger.TButton', background=[('active', "#c0392b")])
        
        self.style.configure('Secondary.TButton', background=BORDER_COLOR, foreground=TEXT_COLOR)
        self.style.map('Secondary.TButton', background=[('active', CARD_BG)])
        
        # Labels
        self.style.configure('TLabel', background=BG_COLOR, foreground=TEXT_COLOR)
        self.style.configure('Sub.TLabel', background=BG_COLOR, foreground=MUTED_TEXT, font=('Segoe UI', 9))
        self.style.configure('Card.TLabel', background=CARD_BG, foreground=TEXT_COLOR)
        self.style.configure('CardSub.TLabel', background=CARD_BG, foreground=MUTED_TEXT, font=('Segoe UI', 9))
        self.style.configure('Header.TLabel', background=BG_COLOR, foreground=TEXT_COLOR, font=('Segoe UI', 14, 'bold'))
        
        # Inputs (Combobox)
        self.style.configure('TCombobox', fieldbackground=BORDER_COLOR, background=BORDER_COLOR, 
                             foreground=TEXT_COLOR, borderwidth=0)
        self.style.map('TCombobox', fieldbackground=[('readonly', BORDER_COLOR)])
        
        # Progress Bar
        self.style.configure('TProgressbar', background=ACCENT_COLOR, troughcolor=BORDER_COLOR, borderwidth=0)

    # --- SPLASH & MODEL DOWNLOADING SCREEN ---
    def show_splash_screen(self):
        self.splash_frame = ttk.Frame(self.root)
        self.splash_frame.pack(expand=True, fill=tk.BOTH, padx=50, pady=50)
        
        title = ttk.Label(self.splash_frame, text="FACE ANALYZER PRO", 
                          font=('Segoe UI', 24, 'bold'), foreground=ACCENT_COLOR)
        title.pack(pady=(100, 10))
        
        subtitle = ttk.Label(self.splash_frame, text="Preparing deep learning models & initialization...", 
                             style='Sub.TLabel', font=('Segoe UI', 12))
        subtitle.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(self.splash_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress_bar.pack(pady=30)
        
        self.status_lbl = ttk.Label(self.splash_frame, text="Checking model files...", style='Sub.TLabel')
        self.status_lbl.pack()
        
        # Start download validation thread
        threading.Thread(target=self.init_models_thread, daemon=True).start()

    def init_models_thread(self):
        try:
            # 1. Download models if needed
            self.update_splash_status("Downloading models (ResNet-10 SSD, YuNet, SFace)...", 20)
            
            # Direct intercept of stdout to read download details isn't required,
            # we will just run the model_downloader
            download_all_models()
            
            self.update_splash_status("Loading models into memory...", 70)
            
            # 2. Instantiate pipeline
            self.pipeline = FacePipeline()
            
            self.update_splash_status("Ready!", 100)
            time.sleep(0.5)
            
            # 3. Transition to main interface
            self.root.after(0, self.load_main_ui)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Initialization Error", 
                f"Failed to download or load models. Please check your internet connection.\n\nError: {e}"))
            self.root.after(0, self.root.destroy)

    def update_splash_status(self, text, percent):
        self.root.after(0, lambda: self.status_lbl.configure(text=text))
        self.root.after(0, lambda: self.progress_bar.configure(value=percent))

    # --- MAIN INTERFACE ---
    def load_main_ui(self):
        # Destroy splash screen
        self.splash_frame.destroy()
        
        # Build Sidebar
        self.create_sidebar()
        
        # Build Main Frame with Tabs
        self.create_main_tabs()
        
        # Start GUI background poll queue
        self.poll_frame_queue()
        
    def create_sidebar(self):
        sidebar = ttk.Frame(self.root, width=320, padding=15)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # Title Card
        title_frame = ttk.Frame(sidebar)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(title_frame, text="Face Analyzer Pro", font=('Segoe UI', 16, 'bold'), foreground=ACCENT_COLOR).pack(anchor='w')
        ttk.Label(title_frame, text="Version 1.0 (Deep Learning)", style='Sub.TLabel').pack(anchor='w')
        
        # Input Source Card
        source_card = ttk.Frame(sidebar, style='Card.TFrame', padding=10)
        source_card.pack(fill=tk.X, pady=(0, 15))
        ttk.Label(source_card, text="INPUT SOURCE", style='Card.TLabel', font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(0, 6))
        
        self.source_var = tk.StringVar(value="Webcam")
        src_webcam = tk.Radiobutton(source_card, text="Live Webcam Feed", variable=self.source_var, value="Webcam",
                                    bg=CARD_BG, fg=TEXT_COLOR, selectcolor=CARD_BG, activebackground=CARD_BG,
                                    activeforeground=TEXT_COLOR, command=self.on_source_change)
        src_webcam.pack(anchor='w', pady=2)
        
        src_file = tk.Radiobutton(source_card, text="Static File (Image/Video)", variable=self.source_var, value="File",
                                  bg=CARD_BG, fg=TEXT_COLOR, selectcolor=CARD_BG, activebackground=CARD_BG,
                                  activeforeground=TEXT_COLOR, command=self.on_source_change)
        src_file.pack(anchor='w', pady=2)
        
        # Model Configuration Card
        model_card = ttk.Frame(sidebar, style='Card.TFrame', padding=10)
        model_card.pack(fill=tk.X, pady=(0, 15))
        ttk.Label(model_card, text="MODEL CONFIGURATION", style='Card.TLabel', font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(0, 8))
        
        # Face Detector
        ttk.Label(model_card, text="Face Detector:", style='CardSub.TLabel').pack(anchor='w')
        self.detector_cb = ttk.Combobox(model_card, values=["SSD DNN", "YuNet", "Haar Cascade"], state="readonly")
        self.detector_cb.current(0)
        self.detector_cb.pack(fill=tk.X, pady=(2, 8))
        
        # Face Recognizer
        ttk.Label(model_card, text="Face Recognizer:", style='CardSub.TLabel').pack(anchor='w')
        self.recognizer_cb = ttk.Combobox(model_card, values=["SFace DNN", "PyTorch Siamese"], state="readonly")
        self.recognizer_cb.current(0)
        self.recognizer_cb.pack(fill=tk.X, pady=(2, 8))
        
        # Thresholds Card
        thresh_card = ttk.Frame(sidebar, style='Card.TFrame', padding=10)
        thresh_card.pack(fill=tk.X, pady=(0, 15))
        ttk.Label(thresh_card, text="LATENCY & THRESHOLDS", style='Card.TLabel', font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(0, 8))
        
        # Detection Threshold
        ttk.Label(thresh_card, text="Detector Confidence:", style='CardSub.TLabel').pack(anchor='w')
        self.det_thresh_var = tk.DoubleVar(value=0.50)
        det_scale = tk.Scale(thresh_card, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                             variable=self.det_thresh_var, bg=CARD_BG, fg=TEXT_COLOR, highlightthickness=0,
                             troughcolor=BORDER_COLOR, activebackground=ACCENT_COLOR)
        det_scale.pack(fill=tk.X, pady=(2, 8))
        
        # Recognition Threshold
        ttk.Label(thresh_card, text="Recognizer Match Limit:", style='CardSub.TLabel').pack(anchor='w')
        self.rec_thresh_var = tk.DoubleVar(value=0.40)
        self.rec_scale = tk.Scale(thresh_card, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                             variable=self.rec_thresh_var, bg=CARD_BG, fg=TEXT_COLOR, highlightthickness=0,
                             troughcolor=BORDER_COLOR, activebackground=ACCENT_COLOR)
        self.rec_scale.pack(fill=tk.X, pady=2)
        
        # Quick Threshold Helper text
        self.thresh_help = ttk.Label(thresh_card, text="Cosine Similarity Match: (Default: 0.40)", style='CardSub.TLabel')
        self.thresh_help.pack(anchor='w', pady=(4, 0))
        
        self.recognizer_cb.bind("<<ComboboxSelected>>", self.on_recognizer_select)
        
        # Primary Action Panel
        self.action_card = ttk.Frame(sidebar, style='Card.TFrame', padding=10)
        self.action_card.pack(fill=tk.X, pady=(0, 15))
        ttk.Label(self.action_card, text="CONTROLS", style='Card.TLabel', font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(0, 8))
        
        # Feed Control Button (Start/Stop)
        self.feed_btn = ttk.Button(self.action_card, text="Start Webcam", style='Success.TButton', command=self.toggle_webcam)
        self.feed_btn.pack(fill=tk.X, pady=4)
        
        # File selector (Webcam / File dependent)
        self.file_btn = ttk.Button(self.action_card, text="Load Image / Video", style='TButton', command=self.load_file)
        self.file_btn.pack(fill=tk.X, pady=4)
        self.file_btn.pack_forget() # Initially hidden since source is Webcam
        
        # Snapshot capture button
        self.snap_btn = ttk.Button(self.action_card, text="Register User (Webcam)", style='Secondary.TButton', command=self.quick_register)
        self.snap_btn.pack(fill=tk.X, pady=4)
        
        # Stats Card
        stats_card = ttk.Frame(sidebar, style='Card.TFrame', padding=10)
        stats_card.pack(fill=tk.BOTH, expand=True)
        ttk.Label(stats_card, text="PERFORMANCE METRICS", style='Card.TLabel', font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(0, 8))
        
        self.stat_fps = ttk.Label(stats_card, text="Frame Rate: N/A", style='CardSub.TLabel')
        self.stat_fps.pack(anchor='w', pady=2)
        self.stat_faces = ttk.Label(stats_card, text="Faces Detected: 0", style='CardSub.TLabel')
        self.stat_faces.pack(anchor='w', pady=2)
        
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.stat_device = ttk.Label(stats_card, text=f"Computing Device: {device_name}", style='CardSub.TLabel')
        self.stat_device.pack(anchor='w', pady=2)
        
        self.stat_time = ttk.Label(stats_card, text="Latency: N/A", style='CardSub.TLabel')
        self.stat_time.pack(anchor='w', pady=2)

    def create_main_tabs(self):
        # Notebook setup
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # TAB 1: Live Feed Workspace
        self.feed_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.feed_tab, text="WORKSPACE VIEWER")
        
        # Video Display Canvas
        self.canvas_frame = tk.Frame(self.feed_tab, borderwidth=1, relief='solid', bg="#050505")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#050505", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        self.status_bar = ttk.Label(self.feed_tab, text="Status: Ready", style='Sub.TLabel', padding=5)
        self.status_bar.pack(anchor='w')
        
        # TAB 2: Database Manager
        self.db_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.db_tab, text="DATABASE MANAGER")
        
        self.create_db_manager_ui()
        
    def create_db_manager_ui(self):
        # Master Layout: Left side listbox, Right side user details
        left_frame = ttk.Frame(self.db_tab, width=220, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)
        
        ttk.Label(left_frame, text="REGISTERED USERS", font=('Segoe UI', 9, 'bold')).pack(anchor='w', pady=(0, 6))
        
        # Scrollbar and Listbox
        scroll = tk.Scrollbar(left_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.db_listbox = tk.Listbox(left_frame, bg=CARD_BG, fg=TEXT_COLOR, selectbackground=ACCENT_COLOR,
                                     selectforeground=TEXT_COLOR, bd=0, highlightthickness=1,
                                     highlightcolor=BORDER_COLOR, highlightbackground=BORDER_COLOR,
                                     yscrollcommand=scroll.set, font=('Segoe UI', 10))
        self.db_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.db_listbox.yview)
        
        self.db_listbox.bind("<<ListboxSelect>>", self.on_db_select)
        
        # Right frame details
        right_frame = ttk.Frame(self.db_tab, padding=15)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.db_user_header = ttk.Label(right_frame, text="Select a user to view templates", style='Header.TLabel')
        self.db_user_header.pack(anchor='w', pady=(0, 10))
        
        self.db_user_meta = ttk.Label(right_frame, text="", style='Sub.TLabel')
        self.db_user_meta.pack(anchor='w', pady=(0, 20))
        
        # Grid area for user registered faces
        self.db_images_frame = ttk.Frame(right_frame, style='Card.TFrame', padding=10)
        self.db_images_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Action buttons
        db_actions = ttk.Frame(right_frame)
        db_actions.pack(fill=tk.X)
        
        register_file_btn = ttk.Button(db_actions, text="Register From Image File", style='Success.TButton', command=self.register_from_file)
        register_file_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        delete_btn = ttk.Button(db_actions, text="Delete Registered User", style='Danger.TButton', command=self.delete_selected_user)
        delete_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        rebuild_btn = ttk.Button(db_actions, text="Force DB Sync", style='Secondary.TButton', command=self.rebuild_db)
        rebuild_btn.pack(side=tk.RIGHT)
        
        self.populate_db_list()

    # --- ACTION HANDLERS ---
    
    def on_recognizer_select(self, event):
        rec_model = self.recognizer_cb.get()
        if rec_model == "SFace DNN":
            self.rec_thresh_var.set(0.40)
            self.thresh_help.configure(text="Cosine Similarity Match: (Default: 0.40)")
        else: # PyTorch Siamese
            self.rec_thresh_var.set(0.60)
            self.thresh_help.configure(text="Cosine Similarity Match: (Default: 0.60)")
            
    def on_source_change(self):
        source = self.source_var.get()
        if source == "Webcam":
            self.file_btn.pack_forget()
            self.feed_btn.pack(fill=tk.X, pady=4)
            self.snap_btn.pack(fill=tk.X, pady=4)
            if self.webcam_active:
                self.status_bar.configure(text="Status: Streaming webcam feed.")
            else:
                self.status_bar.configure(text="Status: Webcam idle.")
        else: # File Source
            self.toggle_webcam(force_stop=True)
            self.feed_btn.pack_forget()
            self.snap_btn.pack_forget()
            self.file_btn.pack(fill=tk.X, pady=4)
            self.status_bar.configure(text="Status: Static file mode active. Load an image or video.")
            self.canvas.delete("all")
            
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image or Video File",
            filetypes=[("Media Files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mkv *.mov"), 
                       ("Images", "*.jpg *.jpeg *.png"), 
                       ("Videos", "*.mp4 *.avi *.mkv *.mov")]
        )
        if not file_path:
            return
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            self.status_bar.configure(text=f"Status: Analyzing image file {os.path.basename(file_path)}...")
            self.process_static_image(file_path)
        elif ext in ['.mp4', '.avi', '.mkv', '.mov']:
            self.status_bar.configure(text=f"Status: Ready to process video file.")
            self.process_video_dialog(file_path)

    def process_static_image(self, file_path):
        # Load image
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Could not load image file.")
            return
            
        t1 = time.time()
        # Run recognition
        results = self.pipeline.process_and_recognize(
            img,
            detector_name=self.detector_cb.get(),
            recognizer_name=self.recognizer_cb.get(),
            conf_threshold=self.det_thresh_var.get(),
            similarity_threshold=self.rec_thresh_var.get()
        )
        latency = (time.time() - t1) * 1000
        
        # Annotate
        annotated = utils.draw_face_annotations(img, results)
        
        # Display
        self.display_static_frame(annotated)
        
        # Update metrics
        self.stat_faces.configure(text=f"Faces Detected: {len(results)}")
        self.stat_time.configure(text=f"Latency: {latency:.1f} ms")
        self.stat_fps.configure(text="Frame Rate: N/A (Static Image)")
        self.status_bar.configure(text=f"Status: Analysis completed in {latency:.1f}ms. Faces detected: {len(results)}.")
        
        # Offer option to save annotated image
        if messagebox.askyesno("Save Image", "Do you want to save the annotated image output?"):
            save_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")]
            )
            if save_path:
                cv2.imwrite(save_path, annotated)
                self.status_bar.configure(text=f"Status: Saved annotated image to {os.path.basename(save_path)}")

    def process_video_dialog(self, file_path):
        if not messagebox.askyesno("Process Video", 
                                    f"Do you want to run face analysis on video: {os.path.basename(file_path)}?\n\nThis will process the video and save a fully annotated output copy."):
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Save Annotated Video Copy As",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")]
        )
        if not save_path:
            return
            
        # Modal progress popup
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Processing Video...")
        progress_win.geometry("450x180")
        progress_win.configure(bg=CARD_BG)
        progress_win.transient(self.root)
        progress_win.grab_set()
        
        ttk.Label(progress_win, text="Processing Video File...", font=('Segoe UI', 12, 'bold'), background=CARD_BG).pack(pady=(20, 10))
        prog_bar = ttk.Progressbar(progress_win, orient=tk.HORIZONTAL, length=350, mode='determinate')
        prog_bar.pack(pady=10)
        
        prog_lbl = ttk.Label(progress_win, text="Reading frames...", background=CARD_BG, style='Sub.TLabel')
        prog_lbl.pack(pady=5)
        
        self.cancel_video_proc = False
        cancel_btn = ttk.Button(progress_win, text="Cancel", style='Danger.TButton', command=lambda: self.cancel_video(progress_win))
        cancel_btn.pack(pady=10)
        
        detector = self.detector_cb.get()
        recognizer = self.recognizer_cb.get()
        det_t = self.det_thresh_var.get()
        rec_t = self.rec_thresh_var.get()
        
        def video_work():
            def cb(curr, total):
                if self.cancel_video_proc:
                    return False # Stop processing
                percent = int((curr / total) * 100)
                progress_win.after(0, lambda: prog_bar.configure(value=percent))
                progress_win.after(0, lambda: prog_lbl.configure(text=f"Processing frame {curr} / {total} ({percent}%)"))
                return True
                
            try:
                t1 = time.time()
                utils.process_video_file(
                    self.pipeline, file_path, save_path,
                    detector_name=detector, recognizer_name=recognizer,
                    conf_thresh=det_t, sim_thresh=rec_t, progress_callback=cb
                )
                duration = time.time() - t1
                if not self.cancel_video_proc:
                    progress_win.after(0, progress_win.destroy)
                    messagebox.showinfo("Video Processed", 
                                        f"Video processed successfully in {duration:.1f}s.\n\nAnnotated copy saved to:\n{save_path}")
            except Exception as ex:
                progress_win.after(0, progress_win.destroy)
                messagebox.showerror("Video Error", f"Failed to process video: {ex}")
                
        threading.Thread(target=video_work, daemon=True).start()

    def cancel_video(self, win):
        self.cancel_video_proc = True
        win.destroy()
        self.status_bar.configure(text="Status: Video processing cancelled.")

    # --- WEBCAM STREAMING (ASYNC THREAD + QUEUE) ---
    
    def toggle_webcam(self, force_stop=False):
        if self.webcam_active or force_stop:
            # Stop
            self.webcam_active = False
            self.feed_btn.configure(text="Start Webcam", style='Success.TButton')
            self.status_bar.configure(text="Status: Webcam idle.")
            
            # Wait for thread termination
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=1.0)
                
            if self.cap:
                self.cap.release()
                self.cap = None
            self.canvas.delete("all")
            self.stat_fps.configure(text="Frame Rate: N/A")
            self.stat_faces.configure(text="Faces Detected: 0")
            self.stat_time.configure(text="Latency: N/A")
        else:
            # Start
            try:
                self.cap = cv2.VideoCapture(self.webcam_source)
                if not self.cap.isOpened():
                    raise RuntimeError("Cannot open webcam source.")
                    
                # Standard resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                self.webcam_active = True
                self.feed_btn.configure(text="Stop Webcam", style='Danger.TButton')
                self.status_bar.configure(text="Status: Stream started. Initializing processing worker...")
                
                # Clear queue
                while not self.frame_queue.empty():
                    self.frame_queue.get()
                    
                # Start background processing worker thread
                self.worker_thread = threading.Thread(target=self.webcam_worker, daemon=True)
                self.worker_thread.start()
            except Exception as e:
                messagebox.showerror("Camera Error", f"Could not connect to webcam index {self.webcam_source}.\n\nError: {e}")
                self.webcam_active = False
                if self.cap:
                    self.cap.release()
                    self.cap = None

    def webcam_worker(self):
        """Asynchronous face worker thread. Pulls raw camera frames, processes faces, pushes to GUI queue."""
        fps_accum = 0
        fps_count = 0
        fps_time = time.time()
        
        while self.webcam_active:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Flip horizontally for mirrored view
            frame = cv2.flip(frame, 1)
            
            t1 = time.time()
            # Run Face Pipeline
            results = self.pipeline.process_and_recognize(
                frame,
                detector_name=self.detector_cb.get(),
                recognizer_name=self.recognizer_cb.get(),
                conf_threshold=self.det_thresh_var.get(),
                similarity_threshold=self.rec_thresh_var.get()
            )
            latency = (time.time() - t1) * 1000
            
            # Annotate
            annotated = utils.draw_face_annotations(frame, results)
            
            # Handle FPS tracking
            fps_count += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps_accum = fps_count / (now - fps_time)
                fps_count = 0
                fps_time = now
                
            # Push results and frame to queue
            data = {
                'frame': annotated,
                'faces': len(results),
                'latency': latency,
                'fps': fps_accum
            }
            
            try:
                # Use put_nowait or replace item to keep queue size minimal and fast
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(data)
            except queue.Full:
                pass
            except queue.Empty:
                pass
                
        if self.cap:
            self.cap.release()

    def poll_frame_queue(self):
        """Check queue every 15ms and draw active frames onto canvas"""
        if self.webcam_active:
            try:
                # Fetch up to latest data, clear stale frames
                data = None
                while not self.frame_queue.empty():
                    data = self.frame_queue.get_nowait()
                    
                if data is not None:
                    # Draw
                    self.display_camera_frame(data['frame'])
                    # Stats
                    self.stat_faces.configure(text=f"Faces Detected: {data['faces']}")
                    self.stat_time.configure(text=f"Latency: {data['latency']:.1f} ms")
                    self.stat_fps.configure(text=f"Frame Rate: {data['fps']:.1f} FPS")
                    self.status_bar.configure(text=f"Status: Streaming at {data['fps']:.1f} FPS (Latency: {data['latency']:.1f}ms)")
            except queue.Empty:
                pass
                
        # Loop every 15ms
        self.root.after(15, self.poll_frame_queue)

    # --- CANVAS RENDERING (WITH ASPECT RATIO RETENTION) ---
    
    def on_canvas_resize(self, event):
        self.canvas.delete("all")
        
    def display_camera_frame(self, frame_bgr):
        self.display_frame_internal(frame_bgr)
        
    def display_static_frame(self, frame_bgr):
        self.display_frame_internal(frame_bgr)

    def display_frame_internal(self, frame_bgr):
        # Convert BGR (OpenCV) to RGB (Tkinter/PIL)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Calculate scaling to fit frame inside Canvas keeping aspect ratio
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return
            
        img_h, img_w = frame_rgb.shape[:2]
        
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        pil_img = Image.fromarray(frame_rgb)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Draw on Canvas centered
        self.photo = ImageTk.PhotoImage(image=pil_img)
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        
        self.canvas.delete("all")
        self.canvas.create_image(x_offset, y_offset, image=self.photo, anchor='nw')

    # --- DATABASE & REGISTRATION MANAGEMENT ---
    
    def populate_db_list(self):
        self.db_listbox.delete(0, tk.END)
        if self.pipeline and self.pipeline.database:
            for name in sorted(self.pipeline.database.keys()):
                self.db_listbox.insert(tk.END, name)
                
    def on_db_select(self, event):
        selection = self.db_listbox.curselection()
        if not selection:
            return
            
        name = self.db_listbox.get(selection[0])
        user_data = self.pipeline.database.get(name, {})
        
        sface_count = len(user_data.get('sface_embeddings', []))
        pytorch_count = len(user_data.get('pytorch_embeddings', []))
        
        self.db_user_header.configure(text=f"User: {name}")
        self.db_user_meta.configure(text=f"SFace Encodings: {sface_count}  |  PyTorch Encodings: {pytorch_count}")
        
        # Render thumbnails of this user's registered faces
        for widget in self.db_images_frame.winfo_children():
            widget.destroy()
            
        person_dir = os.path.join(self.pipeline.known_faces_dir, name)
        if not os.path.exists(person_dir):
            return
            
        # Display up to 6 thumbnails of registered faces
        images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Keep photo references so garbage collector doesn't clear them
        self.thumb_photos = []
        
        for i, img_name in enumerate(images[:6]):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Face crop preview
            faces = self.pipeline.detect_faces_internal(img)
            if faces:
                x, y, w, h = map(int, faces[0][0:4])
                # Clip box
                img_h, img_w = img.shape[:2]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img_w, x + w)
                y2 = min(img_h, y + h)
                face_crop = img[y1:y2, x1:x2]
            else:
                face_crop = img
                
            if face_crop.size == 0:
                continue
                
            # Resize for thumbnail
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_crop).resize((90, 90), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_img)
            self.thumb_photos.append(photo)
            
            # Pack inside grid
            lbl = tk.Label(self.db_images_frame, image=photo, bg=CARD_BG, borderwidth=1, relief='solid')
            lbl.grid(row=i // 3, column=i % 3, padx=10, pady=10)

    def rebuild_db(self):
        self.status_bar.configure(text="Status: Syncing database descriptors...")
        self.db_user_header.configure(text="Processing database sync...")
        self.db_user_meta.configure(text="")
        for widget in self.db_images_frame.winfo_children():
            widget.destroy()
            
        def work():
            self.pipeline.rebuild_database()
            self.root.after(0, self.on_rebuild_complete)
            
        threading.Thread(target=work, daemon=True).start()
        
    def on_rebuild_complete(self):
        self.populate_db_list()
        self.status_bar.configure(text="Status: Database synced.")
        messagebox.showinfo("Sync Complete", "Successfully rebuilt and synced the facial templates database.")

    def delete_selected_user(self):
        selection = self.db_listbox.curselection()
        if not selection:
            messagebox.showwarning("Delete User", "Please select a user to delete from the list.")
            return
            
        name = self.db_listbox.get(selection[0])
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to permanently delete user: '{name}' and all associated template images?"):
            self.pipeline.delete_user(name)
            self.populate_db_list()
            self.db_user_header.configure(text="Select a user to view templates")
            self.db_user_meta.configure(text="")
            for widget in self.db_images_frame.winfo_children():
                widget.destroy()
            self.status_bar.configure(text=f"Status: Deleted user {name}.")

    def register_from_file(self):
        name = simpledialog.askstring("Register Face", "Enter unique username to register:")
        if not name:
            return
            
        name = name.strip()
        if not name.isalnum():
            messagebox.showerror("Error", "Username must be alphanumeric.")
            return
            
        file_path = filedialog.askopenfilename(
            title=f"Select Face Image for {name}",
            filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return
            
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Could not load selected image.")
            return
            
        success, msg = self.pipeline.register_user(name, img)
        if success:
            messagebox.showinfo("Registration Successful", msg)
            self.populate_db_list()
            self.status_bar.configure(text=f"Status: User {name} registered.")
        else:
            messagebox.showerror("Registration Failed", msg)

    def quick_register(self):
        """Webcam instant registration handler"""
        if not self.webcam_active or not self.cap:
            messagebox.showwarning("Webcam Required", "Please start the Live Webcam Feed before capturing a face sample.")
            return
            
        name = simpledialog.askstring("Register Face via Webcam", "Enter username to register:")
        if not name:
            return
            
        name = name.strip()
        if not name.isalnum():
            messagebox.showerror("Error", "Username must be alphanumeric.")
            return
            
        # Freeze camera snapshot
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture snapshot from webcam.")
            return
            
        # Flip frame to align with mirrored view
        frame = cv2.flip(frame, 1)
        
        success, msg = self.pipeline.register_user(name, frame)
        if success:
            messagebox.showinfo("Registration Successful", msg)
            self.populate_db_list()
            self.status_bar.configure(text=f"Status: User {name} registered via snapshot.")
        else:
            messagebox.showerror("Registration Failed", msg)


if __name__ == '__main__':
    # Force output flushing for real-time tracking logs
    sys.stdout.reconfigure(line_buffering=True)
    
    root = tk.Tk()
    app = FaceApp(root)
    
    def on_closing():
        app.toggle_webcam(force_stop=True)
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
