import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=100)

def is_hand_above_face(hand_landmarks, face_y_normalized):
    """Check if the hand is above the detected face."""
    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    return wrist_y < face_y_normalized  # Hand should be higher (lower y value)

def is_open_palm(hand_landmarks):
    """Check if the hand is an open palm (somewhat open, not fully closed)."""
    landmarks = hand_landmarks.landmark
    return (
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y and
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
    )  # Ensuring fingers are slightly open

def detect_hands(frame, rgb_frame):
    """Detect hands in the frame and return hand landmarks."""
    hand_results = hands.process(rgb_frame)
    return hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else []