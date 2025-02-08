import cv2
import mediapipe as mp

# Initialize MediaPipe Models
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Storage for Tracking Raised Hands
persistent_people = {}
assigned_numbers = {}
face_images = {}
current_number = 1

def detect_faces_and_hands(frame, rgb_frame):
    """Detects faces, hands, and upper body joints."""
    results = holistic.process(rgb_frame)
    faces_detected = []
    hands_detected = {}
    landmarks_detected = {}

    # Detect one dot per face (nose landmark for accuracy)
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
        face_x, face_y = int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0])
        faces_detected.append((face_x, face_y))

    # Detect hands
    if results.right_hand_landmarks:
        hands_detected["right"] = results.right_hand_landmarks
    if results.left_hand_landmarks:
        hands_detected["left"] = results.left_hand_landmarks

    # Detect body parts (elbow, shoulder, neck)
    if results.pose_landmarks:
        landmarks_detected["left_shoulder"] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        landmarks_detected["right_shoulder"] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        landmarks_detected["left_elbow"] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW]
        landmarks_detected["right_elbow"] = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]

    return faces_detected, hands_detected, landmarks_detected

def is_hand_raised(hand_landmarks, elbow_landmark):
    """Checks if the hand is above the elbow."""
    return hand_landmarks.landmark[0].y < elbow_landmark.y

def add_question(face_coords, frame):
    """Adds a new person to the queue if not already tracked."""
    global current_number

    fx, fy = face_coords
    face_crop = frame[max(0, fy-50):fy+50, max(0, fx-50):fx+50]

    # Check if this face is already tracked
    for person_id, data in persistent_people.items():
        if data['coords'] == face_coords:
            return  # Avoid duplicate entries

    person_id = current_number
    persistent_people[person_id] = {'coords': face_coords, 'crop': face_crop}
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

        if row:
            # Handle remaining faces that don’t form a full row
            if len(row) == 1:
                row.append(cv2.cvtColor(row[0], cv2.COLOR_BGR2GRAY))  # Add dummy image if only one face
            rows.append(cv2.hconcat(row))

        if rows:
            face_display = rows[0] if len(rows) == 1 else cv2.vconcat(rows)
            cv2.imshow("Raised Hand Order (Professor View)", face_display)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces_detected, hands_detected, landmarks_detected = detect_faces_and_hands(frame, rgb_frame)
    active_people = set()

    # Draw detections
    for face_coords in faces_detected:
        fx, fy = face_coords

        # Draw one dot per face
        cv2.circle(frame, (fx, fy), 5, (255, 0, 0), -1)

        for hand_side, hand_landmarks in hands_detected.items():
            elbow = landmarks_detected.get(f"{hand_side}_elbow")
            shoulder = landmarks_detected.get(f"{hand_side}_shoulder")

            if elbow and shoulder:
                wrist = hand_landmarks.landmark[0]
                wrist_x = int(wrist.x * frame.shape[1])
                wrist_y = int(wrist.y * frame.shape[0])

                elbow_x = int(elbow.x * frame.shape[1])
                elbow_y = int(elbow.y * frame.shape[0])

                shoulder_x = int(shoulder.x * frame.shape[1])
                shoulder_y = int(shoulder.y * frame.shape[0])

                # Draw connections
                cv2.line(frame, (wrist_x, wrist_y), (elbow_x, elbow_y), (0, 255, 255), 2)
                cv2.line(frame, (elbow_x, elbow_y), (shoulder_x, shoulder_y), (0, 255, 255), 2)
                cv2.line(frame, (shoulder_x, shoulder_y), (fx, fy), (0, 255, 255), 2)

                # If the hand is raised, track the face
                if is_hand_raised(hand_landmarks, elbow):
                    add_question(face_coords, frame)
                    active_people.add(id(face_coords))

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
