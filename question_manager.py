persistent_faces = {}  # Store face bounding boxes mapped to assigned numbers
assigned_numbers = {}  # Store face identifiers with their assigned numbers
face_images = []  # Store cropped face images in order
face_positions = []  # Track positions of faces in "Raised Hand Order" window
face_hashes = set()  # Prevent duplicate faces
current_number = 1  # Track the next number to assign

def add_question(face, frame):
    """Add a new raised hand's face to the queue."""
    global current_number

    fx, fy, fw, fh = face
    face_crop = frame[fy:fy + fh, fx:fx + fw]
    face_hash = hash(face_crop.tobytes()) if face_crop.size > 0 else None

    if face_hash and face_hash not in face_hashes:
        persistent_faces[face] = current_number
        assigned_numbers[current_number] = face
        face_images.append((current_number, face_crop))
        face_positions.append((0, len(face_positions) * 150, 120, 150))  # Track positions
        face_hashes.add(face_hash)
        current_number += 1

def clear_question(index):
    """Remove a specific person's face from the question queue."""
    global face_images, persistent_faces, face_positions
    if index < len(face_images):
        number, _ = face_images.pop(index)
        face_positions.pop(index)
        # Remove number from person's face
        persistent_faces = {k: v for k, v in persistent_faces.items() if v != number}

def clear_all_questions():
    """Remove all faces and numbers from the display."""
    global face_images, persistent_faces, assigned_numbers, face_positions
    face_images.clear()
    face_positions.clear()
    persistent_faces.clear()
    assigned_numbers.clear()