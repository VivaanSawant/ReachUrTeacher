import cv2
import mediapipe as mp
import base64
import time
import sqlite3
from flask import Flask, render_template, Response, request
import numpy as np
from collections import deque

app = Flask(__name__)

START_HAND_TIME = time.time()
HAND_DURATION = 0


# --- Mediapipe Setup ---
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands

# --- Configuration ---
MIN_HAND_CONFIDENCE = 0.8
MIN_FACE_CONFIDENCE = 0.75
HAND_FACE_MAX_DISTANCE_RATIO = 1.5
SNAPSHOT_HISTORY = 5

# --- Database Initialization ---
def init_db():
    conn = sqlite3.connect('faces.db')
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_key TEXT,
            timestamp REAL,
            image BLOB
        )
    ''')
    cur.execute("DELETE FROM faces")
    conn.commit()
    conn.close()

def store_face_snapshot(face_key, face_img):
    ret, buf = cv2.imencode('.jpg', face_img)
    if ret:
        img_bytes = buf.tobytes()
        conn = sqlite3.connect('faces.db')
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO faces (face_key, timestamp, image) VALUES (?, ?, ?)",
            (face_key, time.time(), img_bytes)
        )
        conn.commit()
        conn.close()

# --- Face and Hand Tracking Class ---
class FaceHandTracker:
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=MIN_FACE_CONFIDENCE)
        self.hands = mp_hands.Hands(
            min_detection_confidence=MIN_HAND_CONFIDENCE,
            min_tracking_confidence=0.7,
            max_num_hands=5
        )
        self.tracked_faces = {}
        self.hand_history = deque(maxlen=SNAPSHOT_HISTORY)
        # For tracking hand-raise status:
        self.face_active = {}         # True when the face is currently "active" (hand raised)
        self.face_active_start = {}   # Timestamp when the face became active (for stopwatch)

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
        # Flip the frame (mirror view) and convert color space for processing.
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

        global START_HAND_TIME, HAND_DURATION  # Ensure we modify global vars

        if len(active_hands) == 0:
            if abs(time.time() - START_HAND_TIME > 1):  # Ensure it exists before using
                HAND_DURATION = time.time()
        else:
            START_HAND_TIME = time.time()


        # --- FACE-HAND PAIRING & Stopwatch ---
        current_snapshots = {}
        for (fx, fy, fw, fh) in faces:
            face_center = (fx + fw // 2, fy + fh // 2)
            cv2.circle(frame, face_center, 3, (0, 255, 255), -1)
            face_key = f"{fx}_{fy}_{fw}_{fh}"

            # Check if any hand in recent history is near this face.
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
                # Capture the face image.
                face_img = frame[max(fy, 0):min(fy+fh, frame.shape[0]),
                                 max(fx, 0):min(fx+fw, frame.shape[1])]
                if face_img.size > 0:
                    current_snapshots[face_key] = face_img
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 3)

                    # If this face is not yet active (i.e. hand was just raised), mark active,
                    # record the stopwatch start time, and store a snapshot.
                    if not self.face_active.get(face_key, False):
                        self.face_active[face_key] = True
                        self.face_active_start[face_key] = time.time()
                        store_face_snapshot(face_key, face_img)
                    # If already active, the stopwatch continues (do nothing extra here).
            else:
                # If no hand is near, mark the face as inactive and clear its stopwatch start time.
                self.face_active[face_key] = False
                if face_key in self.face_active_start:
                    del self.face_active_start[face_key]

        self.tracked_faces = current_snapshots
        return frame

# --- Frame Generation for Video Feed ---
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

# --- Routes ---
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
    """
    Returns HTML for each active face with a data-start attribute set to
    the timestamp when the face became active (i.e. when the hand was raised).
    The front-end JavaScript will use this timestamp to calculate the elapsed time.
    """
    html = ""
    for face_key, face_img in tracker.tracked_faces.items():
        # Use the stopwatch start time for this face.
        start_time = time.time() - HAND_DURATION
        ret, buf = cv2.imencode('.jpg', face_img)
        if ret:
            b64_face = base64.b64encode(buf.tobytes()).decode('utf-8')
            html += f'''
            <div class="face-card" data-start="{start_time}" data-facekey="{face_key}">
              <img src="data:image/jpeg;base64,{b64_face}" alt="Detected Face" />
              <div class="stopwatch"></div>
            </div>
            '''
    return html if html else "<div>No active hand-face pairs detected</div>"

@app.route('/face_history')
def face_history():
    conn = sqlite3.connect('faces.db')
    cur = conn.cursor()
    cur.execute("SELECT face_key, timestamp, image FROM faces ORDER BY timestamp DESC")
    rows = cur.fetchall()
    conn.close()

    html = ""
    for face_key, ts, image in rows:
        b64_face = base64.b64encode(image).decode('utf-8')
        formatted_time = time.time() - HAND_DURATION
        html += f'''
            <div class="face-card" data-facekey="{face_key}" style="margin: 10px; display: inline-block;">
                <img src="data:image/jpeg;base64,{b64_face}" alt="Face Snapshot" style="width:100px; height:100px; object-fit: cover;"/>
                <div class="timestamp" style="font-size:0.8rem;">{formatted_time}</div>
            </div>
        '''
    if not html:
        html = "<div>No face snapshots found.</div>"
    return html

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
