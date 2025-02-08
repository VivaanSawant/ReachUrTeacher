import cv2
import mediapipe as mp
import base64
from flask import Flask, render_template, Response
import numpy as np
from collections import deque

app = Flask(__name__)

# --- Mediapipe Setup ---
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands

# --- Configuration ---
MIN_HAND_CONFIDENCE = 0.8
MIN_FACE_CONFIDENCE = 0.75
HAND_FACE_MAX_DISTANCE_RATIO = 1.5  # for allowing hands further away
SNAPSHOT_HISTORY = 5  # frames to maintain "hand presence"

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
        
    def _is_open_hand(self, landmarks):
        # improved open-hand detection
        tips = [
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP
        ]
        wrist_y = landmarks.landmark[mp_hands.HandLandmark.WRIST].y
        return all(landmarks.landmark[tip].y < wrist_y for tip in tips)

    def _get_hand_positions(self, frame, landmarks):
        # get wrist and middle finger positions in pixel coords
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
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # --- HAND DETECTION ---
        active_hands = []
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for landmarks in hand_results.multi_hand_landmarks:
                if self._is_open_hand(landmarks):
                    wrist_pos, middle_pos = self._get_hand_positions(frame, landmarks)
                    active_hands.append((wrist_pos, middle_pos))
                    
                    # Draw the hand landmarks
                    cv2.circle(frame, wrist_pos, 5, (0, 255, 0), -1)  # green
                    cv2.circle(frame, middle_pos, 5, (0, 0, 255), -1) # red
                    cv2.line(frame, wrist_pos, middle_pos, (255, 255, 0), 2)  # light blue line
                    # hand center
                    hand_center = ((wrist_pos[0] + middle_pos[0]) // 2, (wrist_pos[1] + middle_pos[1]) // 2)
                    cv2.circle(frame, hand_center, 5, (255, 255, 255), -1) # white

        # Keep track of last N frames of hands
        self.hand_history.append(active_hands)

        # --- FACE-HAND PAIRING & Snapshot ---
        current_snapshots = {}
        for (fx, fy, fw, fh) in faces:
            face_center = (fx + fw//2, fy + fh//2)
            cv2.circle(frame, face_center, 3, (0, 255, 255), -1)  # face center

            hand_near = False
            for hands in self.hand_history:
                for (wrist, middle) in hands:
                    hand_center = ((wrist[0] + middle[0]) // 2, (wrist[1] + middle[1]) // 2)
                    max_distance = fw * HAND_FACE_MAX_DISTANCE_RATIO
                    distance = np.linalg.norm(np.array(face_center) - np.array(hand_center))
                    
                    # optional debug line
                    cv2.line(frame, face_center, hand_center, (0, 128, 255), 1)
                    
                    if distance < max_distance:
                        hand_near = True
                        break
                if hand_near:
                    break

            if hand_near:
                # Crop face
                face_img = frame[max(fy,0):min(fy+fh, frame.shape[0]),
                                 max(fx,0):min(fx+fw, frame.shape[1])]
                if face_img.size > 0:
                    face_key = f"{fx}_{fy}_{fw}_{fh}"
                    current_snapshots[face_key] = face_img
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0,255,0), 3)

        # Update tracked faces for /faces_data route
        self.tracked_faces = current_snapshots
        return frame

tracker = FaceHandTracker()

def generate_frames():
    cap = cv2.VideoCapture(0)  # corrected to 0, removing stray "ç"
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        processed_frame = tracker.process_frame(frame)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue

        # Return multipart/x-mixed-replace response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() +
               b'\r\n')


# ---------------------------
#         ROUTES
# ---------------------------

@app.route('/')
def index():
    # Landing page with "Get Started" button
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # The main teacher/camera page
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    # Live video stream for <img> or <video> tag
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/faces_data')
def faces_data():
    # Return the queued face snapshots as HTML <img> tags
    html = ""
    for face_key, face_img in tracker.tracked_faces.items():
        ret, buf = cv2.imencode('.jpg', face_img)
        if ret:
            b64_face = base64.b64encode(buf.tobytes()).decode('utf-8')
            html += f'''
            <div class="face-card">
              <img src="data:image/jpeg;base64,{b64_face}" alt="Detected Face" />
            </div>
            '''
    return html if html else "<div>No active hand-face pairs detected</div>"

if __name__ == '__main__':
    app.run(debug=True)
