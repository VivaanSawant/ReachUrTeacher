import cv2

# Store face-hand tracking for multiple people
persistent_people = {}  # Maps person ID to face bounding box
assigned_numbers = {}  # Maps numbers to people
face_images = {}  # Maps person ID to snapshots
face_positions = {}  # Maps person ID to display positions
current_number = 1  # Counter for assigning numbers

def add_question(person_id, frame, faces_detected, pose_landmarks):
    """
    Ensures the correct face is linked to a raised hand using body tracking.
    """
    global current_number

    if person_id in persistent_people:
        return

    closest_face = None
    min_distance = float("inf")

    # Get person's shoulders
    left_shoulder = pose_landmarks.get(11, None)  # Left shoulder
    right_shoulder = pose_landmarks.get(12, None)  # Right shoulder

    if not left_shoulder or not right_shoulder:
        return

    shoulder_x = (left_shoulder[0] + right_shoulder[0]) // 2  # Midpoint between shoulders

    for (fx, fy, fw, fh) in faces_detected:
        face_center_x = fx + fw // 2

        # Ensure face belongs to this person based on shoulder alignment
        if abs(face_center_x - shoulder_x) < fw // 2:
            distance = ((face_center_x - shoulder_x) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_face = (fx, fy, fw, fh)

    if closest_face:
        fx, fy, fw, fh = closest_face
        face_crop = frame[fy:fy + fh, fx:fx + fw]

        # Store only one snapshot per unique person
        persistent_people[person_id] = closest_face
        assigned_numbers[current_number] = closest_face
        face_images[person_id] = (current_number, face_crop)
        face_positions[person_id] = (0, len(face_positions) * 150, 120, 150)
        current_number += 1

def remove_hand(person_id):
    """Removes a face snapshot when a hand is lowered."""
    if person_id in persistent_people:
        del persistent_people[person_id]
        del face_images[person_id]
        del face_positions[person_id]