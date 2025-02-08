import cv2

persistent_people = {}
assigned_numbers = {}
face_images = {}
face_positions = {}
current_number = 1

def calculate_iou(bbox1, bbox2):
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    union_area = (w1 * h1) + (w2 * h2) - inter_area

    return inter_area / union_area if union_area > 0 else 0

def add_question(face_bbox, frame):
    """Adds a new person to the queue if not already tracked."""
    global current_number

    person_id = None
    max_iou = 0.5  
    for pid, data in persistent_people.items():
        stored_bbox = data['bbox']
        iou = calculate_iou(face_bbox, stored_bbox)
        if iou > max_iou:
            max_iou = iou
            person_id = pid

    if person_id is None:
        person_id = current_number
        fx, fy, fw, fh = face_bbox
        face_crop = frame[fy:fy+fh, fx:fx+fw]

        persistent_people[person_id] = {'bbox': face_bbox, 'crop': face_crop}
        assigned_numbers[person_id] = current_number
        face_images[person_id] = face_crop
        face_positions[person_id] = (0, (person_id-1)*150, 120, 150)
        current_number += 1
    else:
        fx, fy, fw, fh = face_bbox
        persistent_people[person_id]['bbox'] = face_bbox
        persistent_people[person_id]['crop'] = frame[fy:fy+fh, fx:fx+fw]

def remove_hand(person_id):
    """Removes a person from the queue if their hand is lowered."""
    for storage in [persistent_people, assigned_numbers, face_images, face_positions]:
        storage.pop(person_id, None)
