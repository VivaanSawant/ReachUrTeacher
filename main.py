import cv2
import mediapipe as mp
import time
import math

# Initialize MediaPipe Hands and Face Detection
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

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

def calculate_distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(rgb_frame)
    face_results = face_detection.process(rgb_frame)

    hand_list = []  # Store (hand_id, timestamp, landmarks)
    faces = []  # Store (face_bbox, center_x, center_y)

    # Detect faces
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * frame.shape[1])
            y = int(bboxC.ymin * frame.shape[0])
            w = int(bboxC.width * frame.shape[1])
            h = int(bboxC.height * frame.shape[0])
            center_x = x + w // 2
            center_y = y + h // 2
            faces.append(((x, y, w, h), center_x, center_y))

    # Detect hands and associate them with faces
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Use the WRIST position as a unique key (approximate hand tracking)
            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1]
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0]
            hand_key = (round(wrist_x, 2), round(wrist_y, 2))  # Round to avoid floating point instability

            # Check if this hand is already assigned an ID
            if hand_key not in hand_timestamps:
                hand_timestamps[hand_key] = (next_hand_id, time.time())  # Assign ID and timestamp
                next_hand_id += 1  # Increment for the next hand

            # Get assigned ID and timestamp
            hand_id, first_seen_time = hand_timestamps[hand_key]

            # Store hand if palm is open
            if is_open_palm(hand_landmarks):
                hand_list.append((hand_id, first_seen_time, hand_landmarks, wrist_x, wrist_y))

    # Sort hands by first detected timestamp (earliest = 1)
    hand_list.sort(key=lambda x: x[1])

    # Display total number of hands
    total_hands = len(hand_list)
    cv2.putText(frame, f"Total Hands: {total_hands}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Associate each hand with the closest face
    for order, (hand_id, _, hand_landmarks, wrist_x, wrist_y) in enumerate(hand_list, start=1):
        # Calculate the closest face for this hand
        closest_face = None
        min_distance = float('inf')

        for (face_bbox, face_center_x, face_center_y) in faces:
            distance = calculate_distance(wrist_x, wrist_y, face_center_x, face_center_y)
            if distance < min_distance:
                min_distance = distance
                closest_face = face_bbox

        # Draw the hand mesh (landmarks and connections)
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Draw the closest face bounding box
        if closest_face:
            fx, fy, fw, fh = closest_face
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

        # Display hand number based on order of raising
        wrist_x = int(wrist_x)
        wrist_y = int(wrist_y)
        cv2.putText(frame, f"Hand #{order}", (wrist_x, wrist_y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show the frame
    cv2.imshow("Hand Mesh and Face Association", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
