import cv2
from face_detection import detect_faces
from hand_tracking import detect_hands, is_open_palm
from question_manager import add_question, remove_hand, persistent_people
from ui_display import display_question_queue

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

    # For each face, check if any hand is above it and open
    for face_bbox in faces_detected:
        fx, fy, fw, fh = face_bbox
        face_top_y = fy
        face_left = fx
        face_right = fx + fw

        for hand_id, hand_landmarks in hands_detected.items():
            # Get wrist position
            wrist = hand_landmarks.landmark[0]  # WRIST landmark
            wrist_x = int(wrist.x * frame.shape[1])
            wrist_y = int(wrist.y * frame.shape[0])

            # Check if wrist is above the face and within horizontal bounds
            if wrist_y < face_top_y and face_left <= wrist_x <= face_right:
                if is_open_palm(hand_landmarks):
                    # Add/update the person with this face
                    add_question(face_bbox, frame)
                    active_people.add(id(face_bbox))  # Track using face_bbox id

    # Remove inactive people
    for tracked_person in list(persistent_people.keys()):
        if tracked_person not in active_people:
            remove_hand(tracked_person)

    display_question_queue()

    cv2.imshow("Hand Raise Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()