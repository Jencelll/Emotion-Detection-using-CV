import cv2
import mediapipe as mp
# from deepface import DeepFace  <-- Moved inside method to avoid startup lag
import numpy as np
import threading
import copy

class VideoCamera(object):
    def __init__(self):
        # Camera initialization (DirectShow fallback)
        self.camera_working = True
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Request HD Resolution (1280x720)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.video.isOpened():
             # Fallback to default
            self.video = cv2.VideoCapture(0)
            # Request HD Resolution
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            if not self.video.isOpened():
                print("Error: Could not start camera.")
                self.camera_working = False
                # raise RuntimeError("Could not start camera.") # Don't crash, show error frame

        # Mediapipe setup (Commented out due to import issues)
        # self.mp_face_detection = mp.solutions.face_detection
        # self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        # Haar Cascade fallback
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Emotion state
        self.current_emotion = "Loading AI..."
        self.emotion_probabilities = {"neutral": 100}
        self.last_frame_for_emotion = None
        self.lock = threading.Lock()
        self.running = True
        self.model_loaded = False
        
        # Frame storage
        self.current_frame = None
        # Initialize encoded_frame with a loading image to avoid hanging
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Loading Camera...", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, jpeg = cv2.imencode('.jpg', blank_frame)
        self.encoded_frame = jpeg.tobytes()
        
        # Start video capture thread
        self.video_thread = threading.Thread(target=self.update_video_loop)
        self.video_thread.daemon = True
        self.video_thread.start()
        
        # Start emotion detection thread
        self.emotion_thread = threading.Thread(target=self.detect_emotion_loop)
        self.emotion_thread.daemon = True
        self.emotion_thread.start()

    def __del__(self):
        self.running = False
        if self.camera_working and self.video.isOpened():
            self.video.release()

    def detect_emotion_loop(self):
        # Lazy import to speed up startup
        try:
            from deepface import DeepFace
            print("DeepFace imported successfully.")
        except ImportError:
            print("Error: DeepFace not installed.")
            return

        while self.running:
            if not self.model_loaded:
                # Warmup / Load model
                try:
                    print("Loading Emotion Model...")
                    # Run on a dummy image to trigger weight loading
                    dummy_img = np.zeros((48, 48, 3), dtype=np.uint8)
                    DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False, silent=True)
                    print("Emotion Model Loaded!")
                    self.model_loaded = True
                    with self.lock:
                        self.current_emotion = "Neutral"
                except Exception as e:
                    print(f"Model loading error: {e}")
                    # Keep trying or break?
                    import time
                    time.sleep(1)
                    continue

            if self.last_frame_for_emotion is not None:
                try:
                    # Create a copy to process
                    with self.lock:
                        frame_to_process = self.last_frame_for_emotion.copy()
                        self.last_frame_for_emotion = None # Clear it so we don't re-process same frame immediately
                    
                    # DeepFace expects RGB
                    rgb_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                    
                    objs = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False, silent=True)
                    
                    if objs:
                        # DeepFace returns a list of dicts
                        result = objs[0]
                        
                        # Convert float32 to float for JSON serialization
                        probs = result['emotion']
                        clean_probs = {k: float(v) for k, v in probs.items()}
                        
                        with self.lock:
                            self.current_emotion = result['dominant_emotion']
                            self.emotion_probabilities = clean_probs
                
                except Exception as e:
                    print(f"Emotion detection error: {e}")
                    pass
            else:
                import time
                time.sleep(0.1) # Sleep briefly to save CPU when no new frame

    def update_video_loop(self):
        read_failures = 0
        while self.running:
            if not self.camera_working:
                # Create a dummy black frame with error text
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera Error", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    with self.lock:
                        self.current_frame = frame
                        self.encoded_frame = jpeg.tobytes()
                        self.current_emotion = "Error"
                        self.emotion_probabilities = {"error": 100}
                
                import time
                time.sleep(1)
                continue

            success, frame = self.video.read()
            if success:
                read_failures = 0
                
                # OPTIMIZATION: Process on a smaller frame for speed
                # We keep 'frame' as the high-quality original (1280x720)
                # We create 'small_frame' for expensive operations like Face Detection
                
                # 1. Lighting Pre-processing (CLAHE)
                # Applying CLAHE to full HD frame can be slow. 
                # Let's try to keep it, but if it lags, we might need to skip it or optimize.
                try:
                    # Convert to LAB color space
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    
                    # Create CLAHE object
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    
                    # Apply CLAHE to L-channel
                    cl = clahe.apply(l)
                    
                    # Merge channels
                    limg = cv2.merge((cl,a,b))
                    
                    # Convert back to BGR
                    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                except:
                    pass

                # 2. Face Detection (Optimized)
                try:
                    # Resize to a smaller width for detection (e.g., 320px)
                    # This drastically reduces detection time (from ~50ms to ~5ms)
                    height, width = frame.shape[:2]
                    target_detection_width = 320
                    scale_factor = width / target_detection_width
                    
                    small_frame = cv2.resize(frame, (target_detection_width, int(height / scale_factor)))
                    gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces on small frame
                    faces_small = self.face_cascade.detectMultiScale(
                        gray_small, 
                        scaleFactor=1.1, 
                        minNeighbors=5, 
                        minSize=(20, 20) # Adjusted for small resolution
                    )
                    
                    face_roi_img = None
                    for (x_small, y_small, w_small, h_small) in faces_small:
                        # Scale coordinates back to original frame size
                        x = int(x_small * scale_factor)
                        y = int(y_small * scale_factor)
                        w = int(w_small * scale_factor)
                        h = int(h_small * scale_factor)
                        
                        # Ensure coordinates are within bounds
                        x = max(0, x)
                        y = max(0, y)
                        w = min(width - x, w)
                        h = min(height - y, h)

                        if w > 0 and h > 0:
                            face_roi_img = frame[y:y+h, x:x+w]
                        
                        # Draw rectangle on original high-res frame
                        # Use a thinner line for HD look
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        with self.lock:
                            emotion_text = self.current_emotion
                        
                        # Add text background for better readability
                        text_size, _ = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                        cv2.rectangle(frame, (x, y - 35), (x + text_size[0], y), (0, 255, 0), -1)
                        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                    
                    # Update emotion thread input
                    self.update_emotion_frame(frame, face_roi_img)
                    
                    # Encode frame
                    # Reduce JPEG quality slightly to 85 (default is 95) to speed up transfer without visible loss
                    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if ret:
                        with self.lock:
                            self.current_frame = frame.copy()
                            self.encoded_frame = jpeg.tobytes()
                except Exception as e:
                    print(f"Frame processing error: {e}")
            else:
                read_failures += 1
                if read_failures > 50: # If 50 frames fail (~1-2 seconds), assume camera died
                    print("Camera failed to read frames. Switching to error mode.")
                    self.camera_working = False
                cv2.waitKey(10)

    def update_emotion_frame(self, frame, face_roi):
        # Update the frame for the emotion thread to pick up
        # We prefer the face ROI if available, but DeepFace might need context? 
        # Actually DeepFace works on face chips.
        with self.lock:
            if face_roi is not None and face_roi.size > 0:
                self.last_frame_for_emotion = face_roi
            else:
                self.last_frame_for_emotion = frame

    def get_frame(self):
        with self.lock:
            if self.encoded_frame is not None:
                return self.encoded_frame
        return None

    # Removed old get_frame logic as it is now in update_video_loop
    # def get_frame(self): ...


    def get_current_emotion_data(self):
        with self.lock:
            return {
                "dominant_emotion": self.current_emotion,
                "probabilities": self.emotion_probabilities
            }

    def get_snapshot(self):
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy(), self.current_emotion
        return None, None
