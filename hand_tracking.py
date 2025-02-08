import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Initialize MediaPipe Hands (Using Improved Hand Tracking from mainVivaan.py)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=5)

def is_open_palm(hand_landmarks):
    """Check if the hand is an open palm (all fingers extended)."""
    landmarks = hand_landmarks.landmark

    # Get y-coordinates of fingertips
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y

    # Get y-coordinates of DIP joints (knuckles below fingertips)
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
    middle_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
    ring_dip = landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y
    pinky_dip = landmarks[mp_hands.HandLandmark.PINKY_DIP].y

    return all(finger_tip < dip for finger_tip, dip in zip(
        [index_tip, middle_tip, ring_tip, pinky_tip],
        [index_dip, middle_dip, ring_dip, pinky_dip]
    ))

def is_hand_above_face(hand_landmarks, face_y_normalized):
    """Check if the hand is above the detected face."""
    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    return wrist_y < face_y_normalized  # Hand should be higher (lower y value)

def detect_faces(frame, rgb_frame):
    """Detect faces in the frame and return bounding boxes."""
    face_results = face_detection.process(rgb_frame)
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

            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return faces_detected

def detect_hands(frame, rgb_frame):
    """Detect hands using improved tracking from mainVivaan.py."""
    hand_results = hands.process(rgb_frame)
    detected_hands = []

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            detected_hands.append(hand_landmarks)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return detected_hands