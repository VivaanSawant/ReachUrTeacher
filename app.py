import cv2
import mediapipe as mp
import base64
import time  # NEW: to track timestamps
from flask import Flask, render_template, Response, request
import numpy as np
from collections import deque

app = Flask(__name__)

# --- Mediapipe Setup ---
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands

# --- Configuration ---
MIN_HAND_CONFIDENCE = 0.8
MIN_FACE_CONFIDENCE = 0.75
HAND_FACE_MAX_DISTANCE_RATIO = 1.5
SNAPSHOT_HISTORY = 5

class FaceHandTracker:
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=MIN_FACE_CONFIDENCE)
        self.hands = mp_hands.Hands(
            min_detection_confidence=MIN_HAND_CONFIDENCE,
            min_tracking_confidence=0.7,
            max_num_hands=2
        )
        self.tracked_faces = {}
        self.hand_history = deque(maxlen=SNAPSHOT_HISTORY)
        self.face_first_seen = {}  # NEW: track when each face-key first appeared

    def _is_open_hand(self, landmarks):
        tips = [
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP
        ]
        wrist_y = landmarks.landmark[mp_hands.HandLandmark.WRIST].y
        return all(landmarks.landmark[tip].y < wrist_y for tip in tips)

    def _get_hand_positions(self, frame, landmarks):
        wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
        middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        return (
            (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])),
            (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0]))
        )

    def process_frame(self, frame):
        # Flip and convert for processing
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- FACE DETECTION ---
        faces = []
        face_results = self.face_detection.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * frame.shape[1])
                y = int(bbox.ymin * frame.shape[0])
                w = int(bbox.width * frame.shape[1])
                h = int(bbox.height * frame.shape[0])
                faces.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # --- HAND DETECTION ---
        active_hands = []
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for landmarks in hand_results.multi_hand_landmarks:
                if self._is_open_hand(landmarks):
                    wrist_pos, middle_pos = self._get_hand_positions(frame, landmarks)
                    active_hands.append((wrist_pos, middle_pos))
                    
                    cv2.circle(frame, wrist_pos, 5, (0, 255, 0), -1)
                    cv2.circle(frame, middle_pos, 5, (0, 0, 255), -1)
                    cv2.line(frame, wrist_pos, middle_pos, (255, 255, 0), 2)

        self.hand_history.append(active_hands)

        # --- FACE-HAND PAIRING & Snapshot ---
        current_snapshots = {}
        for (fx, fy, fw, fh) in faces:
            face_center = (fx + fw//2, fy + fh//2)
            cv2.circle(frame, face_center, 3, (0, 255, 255), -1)

            hand_near = False
            for hands in self.hand_history:
                for (wrist, middle) in hands:
                    hand_center = ((wrist[0] + middle[0]) // 2, (wrist[1] + middle[1]) // 2)
                    max_distance = fw * HAND_FACE_MAX_DISTANCE_RATIO
                    distance = np.linalg.norm(np.array(face_center) - np.array(hand_center))
                    
                    if distance < max_distance:
                        hand_near = True
                        break
                if hand_near:
                    break

            if hand_near:
                face_img = frame[max(fy,0):min(fy+fh, frame.shape[0]),
                                 max(fx,0):min(fx+fw, frame.shape[1])]
                if face_img.size > 0:
                    face_key = f"{fx}_{fy}_{fw}_{fh}"
                    current_snapshots[face_key] = face_img
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0,255,0), 3)

                    # NEW: If it's the first time we've seen this face_key, record the time
                    if face_key not in self.face_first_seen:
                        self.face_first_seen[face_key] = time.time()

        self.tracked_faces = current_snapshots
        return frame

# NOTE: You had two generate_frames definitions; let's unify them properly:
def generate_frames(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        processed_frame = tracker.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() +
               b'\r\n')

tracker = FaceHandTracker()

@app.route('/')
def index():
    return render_template('index.html')

def get_available_cameras():
    cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append({"id": i, "name": f"Camera {i}"})
            cap.release()
    return cameras

@app.route('/dashboard')
def dashboard():
    cameras = get_available_cameras()
    return render_template('dashboard.html', cameras=cameras)

@app.route('/video_feed')
def video_feed():
    camera_id = int(request.args.get('camera_id', 0))
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/faces_data')
def faces_data():
    """Return the queued face snapshots as HTML <div> tags:
       1) data-start holds the epoch time they first appeared
       2) .stopwatch can show the real-time timer in front-end
    """
    html = ""
    for face_key, face_img in tracker.tracked_faces.items():
        start_time = tracker.face_first_seen.get(face_key, time.time())
        ret, buf = cv2.imencode('.jpg', face_img)
        if ret:
            b64_face = base64.b64encode(buf.tobytes()).decode('utf-8')
            html += f'''
            <div class="face-card"
                 data-start="{start_time}"
                 data-facekey="{face_key}">
              <img src="data:image/jpeg;base64,{b64_face}" alt="Detected Face" />
              <!-- We'll add a stopwatch area below the face -->
              <div class="stopwatch"></div>
            </div>
            '''
    return html if html else "<div>No active hand-face pairs detected</div>"

if __name__ == '__main__':
    app.run(debug=True)
