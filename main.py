import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=100)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

hand_timestamps = {}  # Store timestamp for first detection of raised hand
persistent_faces = {}  # Store face bounding boxes mapped to assigned numbers
assigned_numbers = {}  # Store face identifiers with their assigned numbers
face_images = []  # Store cropped face images in order
face_hashes = set()  # Prevent duplicate faces
face_positions = []  # Track positions of faces in "Raised Hand Order" window
current_number = 1  # Track the next number to assign

def is_hand_above_face(hand_landmarks, face_y_normalized):
    """Check if the hand is above the detected face."""
    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    return wrist_y < face_y_normalized  # Hand should be higher (lower y value)

def is_open_palm(hand_landmarks):
    """Check if the hand is an open palm (somewhat open, not fully closed)."""
    landmarks = hand_landmarks.landmark
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y

    return index_tip < index_dip and middle_tip < middle_dip  # Some fingers should be open

def clear_question(index):
    """Remove a specific person's face from the question queue."""
    global face_images, persistent_faces, face_positions
    if index < len(face_images):
        number, _ = face_images.pop(index)
        face_positions.pop(index)
        # Remove number from person's face
        persistent_faces = {k: v for k, v in persistent_faces.items() if v != number}

def clear_all_questions():
    """Remove all faces and numbers from the display."""
    global face_images, persistent_faces, assigned_numbers, face_positions
    face_images.clear()
    face_positions.clear()
    persistent_faces.clear()
    assigned_numbers.clear()

def on_mouse_click(event, x, y, flags, param):
    """Handle mouse click events on the Raised Hand Order window."""
    if event == cv2.EVENT_LBUTTONDOWN:
        for index, (px, py, pw, ph) in enumerate(face_positions):
            if px < x < px + pw and py < y < py + ph:
                print(f"Clicked on face {index + 1}. Removing...")
                clear_question(index)
                return

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)  # Improve face detection for darker complexions
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb_frame)
    face_results = face_detection.process(rgb_frame)

    hands_detected = []
    faces_detected = []

    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = (
                int(bboxC.xmin * frame.shape[1]),
                int(bboxC.ymin * frame.shape[0]),
                int(bboxC.width * frame.shape[1]),
                int(bboxC.height * frame.shape[0])
            )

            y = max(0, y - int(0.3 * h))  # Adjust for hats
            faces_detected.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if hand_results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_open_palm(hand_landmarks):
                for (fx, fy, fw, fh) in faces_detected:
                    face_top_y_normalized = fy / frame.shape[0]  
                    
                    if is_hand_above_face(hand_landmarks, face_top_y_normalized):
                        if idx not in hand_timestamps:
                            hand_timestamps[idx] = time.time()

                            face_crop = frame[fy:fy + fh, fx:fx + fw]
                            face_hash = hash(face_crop.tobytes()) if face_crop.size > 0 else None

                            if face_hash and face_hash not in face_hashes:
                                persistent_faces[(fx, fy, fw, fh)] = current_number
                                assigned_numbers[current_number] = (fx, fy, fw, fh)
                                face_images.append((current_number, face_crop))
                                face_positions.append((0, len(face_positions) * 150, 120, 150))  # Track positions
                                face_hashes.add(face_hash)
                                current_number += 1

    for face, number in persistent_faces.items():
        fx, fy, fw, fh = face
        cv2.putText(frame, f"{number}", (fx + fw // 2, fy - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    if face_images:
        max_faces_per_row = 4
        face_size = 120
        rows = []
        row = []

        for index, (num, face_img) in enumerate(face_images):
            face_resized = cv2.resize(face_img, (face_size, face_size))
            label = np.zeros((30, face_size, 3), dtype=np.uint8)
            cv2.putText(label, f"#{num}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            combined = np.vstack([label, face_resized])
            row.append(combined)

            if len(row) >= max_faces_per_row:
                rows.append(np.hstack(row))
                row = []

        if row:
            rows.append(np.hstack(row))

        face_display = np.vstack(rows)
        cv2.imshow("Raised Hand Order", face_display)
        cv2.setMouseCallback("Raised Hand Order", on_mouse_click)

    cv2.putText(frame, "Press 'C' to Clear All Questions", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Face-Linked Hand Raising Order", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        clear_all_questions()

cap.release()
cv2.destroyAllWindows()