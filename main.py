import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=5)

# Open webcam
cap = cv2.VideoCapture(0)

hand_timestamps = {}  # Track first detection time for each hand

def is_open_palm(hand_landmarks):
    """Check if the hand is an open palm (all fingers extended)."""
    landmarks = hand_landmarks.landmark

    # Get y-coordinates of fingers
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y

    # Get y-coordinates of DIP joints (knuckles just below fingertips)
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
    middle_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
    ring_dip = landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y
    pinky_dip = landmarks[mp_hands.HandLandmark.PINKY_DIP].y

    # If all fingertips are above (y smaller) than DIP joints → Open palm
    if (index_tip < index_dip and
        middle_tip < middle_dip and
        ring_tip < ring_dip and
        pinky_tip < pinky_dip):
        return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_list = []  # Store hand data (ID, timestamp, landmark positions)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if the hand is an open palm
            if is_open_palm(hand_landmarks):
                if idx not in hand_timestamps:
                    hand_timestamps[idx] = time.time()  # Save first detection time
                
                # Store hand info (ID, timestamp, coordinates)
                hand_list.append((idx, hand_timestamps[idx], hand_landmarks.landmark))

    # Sort hands by timestamp (earliest first)
    hand_list.sort(key=lambda x: x[1])

    # Display numbers above hands
    for order, (hand_id, _, landmarks) in enumerate(hand_list, start=1):
        wrist_x = int(landmarks[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
        wrist_y = int(landmarks[mp_hands.HandLandmark.WRIST].y * frame.shape[0])

        # Draw number above hand
        cv2.putText(frame, f"{order}", (wrist_x, wrist_y - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show the frame
    cv2.imshow("Hand Order Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
