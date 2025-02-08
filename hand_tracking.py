import cv2
import mediapipe as mp

# Initialize MediaPipe Hands & Pose Estimation
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=100)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_hands(frame, rgb_frame):
    """Detects hands and assigns unique IDs to them."""
    results = hands.process(rgb_frame)
    detected_hands = {}

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            detected_hands[idx] = hand_landmarks  # Assigns a unique ID

    return detected_hands

def detect_pose(frame, rgb_frame):
    """Detects full-body pose to link hands to faces."""
    results = pose.process(rgb_frame)
    pose_landmarks = {}

    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            pose_landmarks[idx] = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))

    return pose_landmarks

def is_hand_above_face(hand_landmarks, face_y_normalized):
    """Check if the hand is above the detected face."""
    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    return wrist_y < face_y_normalized

def is_open_palm(hand_landmarks):
    """Check if the hand is open."""
    landmarks = hand_landmarks.landmark
    return (
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y and
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
    )