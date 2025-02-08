import cv2
import mediapipe as mp

# Initialize MediaPipe Models
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=100)

# Storage for Tracking Raised Hands
persistent_people = {}
assigned_numbers = {}
face_images = {}
current_number = 1

def detect_faces(frame, rgb_frame):
    """Detects faces in the frame and returns bounding boxes."""
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
            faces_detected.append((x, y, w, h))

    return faces_detected

def detect_hands(frame, rgb_frame):
    """Detects hands in the frame and returns their landmarks."""
    results = hands.process(rgb_frame)
    detected_hands = {}

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            detected_hands[idx] = hand_landmarks

    return detected_hands

def is_somewhat_open_palm(hand_landmarks):
    """Determines if the hand is somewhat open based on finger positions."""
    landmarks = hand_landmarks.landmark
    return (
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.WRIST].y or
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.WRIST].y
    )

def add_question(face_bbox, frame):
    """Adds a new person to the queue if not already tracked."""
    global current_number

    fx, fy, fw, fh = face_bbox
    face_crop = frame[fy:fy+fh, fx:fx+fw]

    # Check if this face is already tracked
    for person_id, data in persistent_people.items():
        if data['bbox'] == face_bbox:
            return  # Avoid duplicate entries

    person_id = current_number
    persistent_people[person_id] = {'bbox': face_bbox, 'crop': face_crop}
    assigned_numbers[person_id] = current_number
    face_images[person_id] = face_crop
    current_number += 1

def remove_hand(person_id):
    """Removes a person from the queue if their hand is lowered."""
    for storage in [persistent_people, assigned_numbers, face_images]:
        storage.pop(person_id, None)

def display_question_queue():
    """Displays faces in order of hand raising."""
    if persistent_people:
        max_faces_per_row = 6
        face_size = 100
        rows = []
        row = []

        for person_id, data in persistent_people.items():
            face_img = data['crop']
            num = assigned_numbers[person_id]

            # Resize face snapshot
            face_resized = cv2.resize(face_img, (face_size, face_size))
            label = face_resized.copy()
            cv2.putText(label, f"#{num}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            row.append(cv2.vconcat([label, face_resized]))

            if len(row) >= max_faces_per_row:
                rows.append(cv2.hconcat(row))
                row = []

        if rows:
            cv2.imshow("Raised Hand Order (Professor View)", cv2.vconcat(rows))

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces_detected = detect_faces(frame, rgb_frame)
    hands_detected = detect_hands(frame, rgb_frame)

    active_people = set()

    # Draw detections
    for face_bbox in faces_detected:
        fx, fy, fw, fh = face_bbox
        face_top_y = fy
        face_bottom_y = fy + fh
        face_left = fx
        face_right = fx + fw

        # Draw face box
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)

        for hand_id, hand_landmarks in hands_detected.items():
            wrist = hand_landmarks.landmark[0]  # WRIST landmark
            index_tip = hand_landmarks.landmark[8]  # Index finger tip
            pinky_tip = hand_landmarks.landmark[20]  # Pinky tip

            wrist_x = int(wrist.x * frame.shape[1])
            wrist_y = int(wrist.y * frame.shape[0])
            index_y = int(index_tip.y * frame.shape[0])
            pinky_y = int(pinky_tip.y * frame.shape[0])

            # Draw hand tracking
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Draw wrist box
            cv2.rectangle(frame, (wrist_x-20, wrist_y-20), (wrist_x+20, wrist_y+20), (0, 255, 0), 2)
            
            # Draw line from face to hand
            cv2.line(frame, (fx + fw//2, fy + fh//2), (wrist_x, wrist_y), (0, 255, 255), 2)

            # If wrist is above the face and fingers are somewhat open, track face
            if (wrist_y < face_top_y and face_left <= wrist_x <= face_right and 
                is_somewhat_open_palm(hand_landmarks) and 
                index_y < face_bottom_y and pinky_y < face_bottom_y):
                
                add_question(face_bbox, frame)
                active_people.add(id(face_bbox))

    # Remove people who no longer have their hands raised
    for tracked_person in list(persistent_people.keys()):
        if tracked_person not in active_people:
            remove_hand(tracked_person)

    display_question_queue()
    cv2.imshow("Hand Raise Tracking (Student View)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
