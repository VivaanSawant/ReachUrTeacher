import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=5)

# Open webcam
cap = cv2.VideoCapture(0)

hand_timestamps = {}  # Dictionary to store first detection time per hand
next_hand_id = 1  # Counter for assigning unique hand IDs

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
    return all(tip < dip for tip, dip in zip(
        [index_tip, middle_tip, ring_tip, pinky_tip],
        [index_dip, middle_dip, ring_dip, pinky_dip]
    ))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_list = []  # Store (hand_id, timestamp, landmarks)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Use the WRIST position as a unique key (approximate hand tracking)
            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            hand_key = (round(wrist_x, 2), round(wrist_y, 2))  # Round to avoid floating point instability

            # Check if this hand is already assigned an ID
            if hand_key not in hand_timestamps:
                hand_timestamps[hand_key] = (next_hand_id, time.time())  # Assign ID and timestamp
                next_hand_id += 1  # Increment for the next hand

            # Get assigned ID and timestamp
            hand_id, first_seen_time = hand_timestamps[hand_key]

            # Store info if hand is open
            if is_open_palm(hand_landmarks):
                hand_list.append((hand_id, first_seen_time, hand_landmarks))

    # Sort hands by first detected timestamp (earliest = 1)
    hand_list.sort(key=lambda x: x[1])

    # Display total number of hands
    total_hands = len(hand_list)
    cv2.putText(frame, f"Total Hands: {total_hands}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw the hand mesh for all detected hands
    for order, (hand_id, _, hand_landmarks) in enumerate(hand_list, start=1):
        # Draw the hand mesh (landmarks and connections)
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Calculate wrist position for displaying the order
        wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
        wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])

        # Display number based on order of raising
        cv2.putText(frame, f"{order}", (wrist_x, wrist_y - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show the frame
    cv2.imshow("Hand Mesh for All Hands", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
