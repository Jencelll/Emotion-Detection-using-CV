from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera
import cv2
import os
import datetime
import threading

app = Flask(__name__)

# Global camera instance
camera = None

def init_camera():
    global camera
    if camera is None:
        print("Initializing camera and models...")
        camera = VideoCamera()
        print("Camera initialized.")

# Initialize in a separate thread to not block Flask startup, 
# but start immediately so it's ready when user arrives
threading.Thread(target=init_camera).start()

def get_camera():
    global camera
    # Wait for initialization if accessed too early
    while camera is None:
        import time
        time.sleep(0.1)
    return camera

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
             # Wait a bit if no frame
             import time
             time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(gen(get_camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_data')
def emotion_data():
    return jsonify(get_camera().get_current_emotion_data())

@app.route('/snapshot', methods=['POST'])
def snapshot():
    frame, emotion = get_camera().get_snapshot()
    if frame is not None:
        if not os.path.exists('snapshots'):
            os.makedirs('snapshots')
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshots/{timestamp}_{emotion}.jpg"
        cv2.imwrite(filename, frame)
        return jsonify({"status": "success", "filename": filename, "emotion": emotion})
    return jsonify({"status": "error", "message": "Could not capture frame"})

if __name__ == '__main__':
    # Use threaded=True to allow multiple requests (video feed + ajax)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
