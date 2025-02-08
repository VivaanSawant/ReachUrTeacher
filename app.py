import cv2
import mediapipe as mp
import base64
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import numpy as np
from collections import deque

app = Flask(__name__, template_folder=".")
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow frontend requests

# MediaPipe Initialization
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands

# Configuration
MIN_HAND_CONFIDENCE = 0.8
MIN_FACE_CONFIDENCE = 0.75
HAND_FACE_MAX_DISTANCE_RATIO = 1.5
SNAPSHOT_HISTORY = 5

class FaceHandTracker:
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=MIN_FACE_CONFIDENCE)
        self.hands = mp_hands.Hands(min_detection_confidence=MIN_HAND_CONFIDENCE, min_tracking_confidence=0.7, max_num_hands=2)
        self.tracked_faces = {}
        self.hand_history = deque(maxlen=SNAPSHOT_HISTORY)

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        faces = []
        face_results = self.face_detection.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x, y, w, h = (
                    int(bbox.xmin * frame.shape[1]),
                    int(bbox.ymin * frame.shape[0]),
                    int(bbox.width * frame.shape[1]),
                    int(bbox.height * frame.shape[0]),
                )
                faces.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Capture hand presence
        self.tracked_faces = {f"{x}_{y}_{w}_{h}": frame[y:y + h, x:x + w] for x, y, w, h in faces}
        return frame

tracker = FaceHandTracker()

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        processed_frame = tracker.process_frame(frame)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classroom')
def classroom():
    return render_template('classroom.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/faces_data')
def faces_data():
    html = ""
    for face_key, face_img in tracker.tracked_faces.items():
        ret, buf = cv2.imencode('.jpg', face_img)
        if ret:
            b64_face = base64.b64encode(buf.tobytes()).decode('utf-8')
            html += f'<div class="face-card"><img src="data:image/jpeg;base64,{b64_face}" alt="Detected Face"></div>'
    return html if html else "<div>No hands raised.</div>"

@app.route('/clear_hands', methods=['POST'])
def clear_hands():
    tracker.tracked_faces = {}
    return jsonify({"message": "Hands cleared."})

if __name__ == '__main__':
    app.run(debug=True)
