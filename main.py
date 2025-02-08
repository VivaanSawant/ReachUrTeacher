import cv2
from face_detection import detect_faces
from hand_tracking import detect_hands, detect_pose, is_open_palm, is_hand_above_face
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
    pose_landmarks = detect_pose(frame, rgb_frame)

    active_people = set()

    for hand_id, hand_landmarks in hands_detected.items():
        if 0 in pose_landmarks and 11 in pose_landmarks and 12 in pose_landmarks:  # Check for key body points
            left_shoulder_x, _ = pose_landmarks[11]  # Left shoulder
            right_shoulder_x, _ = pose_landmarks[12]  # Right shoulder
            shoulder_mid_x = (left_shoulder_x + right_shoulder_x) // 2  # Shoulder center
            person_id = hand_id  # Unique identifier for a person

            for (fx, fy, fw, fh) in faces_detected:
                face_top_y_normalized = fy / frame.shape[0]

                if is_open_palm(hand_landmarks) and is_hand_above_face(hand_landmarks, face_top_y_normalized):
                    face_center_x = fx + fw // 2

                    # Ensure hand belongs to this person by checking alignment with shoulders
                    if abs(face_center_x - shoulder_mid_x) < fw // 2:
                        add_question(person_id, frame, faces_detected, pose_landmarks)
                        active_people.add(person_id)

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