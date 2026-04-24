import cv2
import numpy as np
import mysql.connector
import pickle
import os
from datetime import datetime
from face_processor import FaceProcessor


def get_db_connection():
    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST", "tramway.proxy.rlwy.net"),
        port=int(os.environ.get("MYSQL_PORT", 35641)),
        user=os.environ.get("MYSQL_USER", "root"),
        password=os.environ.get("MYSQL_PASSWORD", "aLgTKfCvYPjSjvsQZHAzQCvKRqikTVCL"),
        database=os.environ.get("MYSQL_DATABASE", "railway")
    )




def load_all_encodings():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, full_name, face_encoding 
            FROM missing_persons 
            WHERE status = 'missing' AND face_encoding IS NOT NULL
        """)
        persons = cursor.fetchall()
        cursor.close()
        conn.close()

        encodings_list = []
        for person in persons:
            encoding = pickle.loads(person['face_encoding'])
            encodings_list.append({
                'id': person['id'],
                'name': person['full_name'],
                'encoding': encoding
            })

        print(f"Loaded {len(encodings_list)} persons from database")
        return encodings_list

    except Exception as e:
        print(f"Error loading encodings: {e}")
        return []


def save_alert(person_id, confidence, snapshot_path, camera_location="Webcam"):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alerts 
            (person_id, camera_location, detected_time, confidence_score, snapshot_path)
            VALUES (%s, %s, %s, %s, %s)
        """, (person_id, camera_location, datetime.now(), confidence, snapshot_path))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Alert saved for person ID: {person_id}")
    except Exception as e:
        print(f"Error saving alert: {e}")


def get_face_key(x, y, w, h, tolerance=60):
    """
    Create a stable key for a face based on its center position.
    Faces within 'tolerance' pixels are considered the same face.
    """
    cx = int(x) + int(w) // 2
    cy = int(y) + int(h) // 2
    # Convert to plain int to avoid np.int32 dict key mismatch
    return (int(cx // tolerance), int(cy // tolerance))


def find_matching_state(face_key, face_states):
    """
    Find an existing state entry close to this face's position key.
    """
    kx, ky = face_key
    for key in face_states:
        ox, oy = key
        if abs(kx - ox) <= 1 and abs(ky - oy) <= 1:
            return key
    return None


def run_camera():
    processor = FaceProcessor()

    print("Loading missing persons from database...")
    known_persons = load_all_encodings()

    if len(known_persons) == 0:
        print("No missing persons in database.")
        return

    print(f"Watching for {len(known_persons)} missing persons...")

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Camera opened! Press Q to quit.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    alert_cooldown = {}
    cooldown_seconds = 30
    frame_count = 0
    reload_every_frames = 100
    stay_green_frames = 30

    # Position-based face state tracking
    # Key: (grid_x, grid_y) based on face center position
    # Value: {'match': person, 'confidence': float, 'frames_left': int}
    face_states = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break

        frame_count += 1

        # Reload DB periodically to pick up new entries
        if frame_count % reload_every_frames == 0:
            print("Refreshing encodings from database...")
            new_persons = load_all_encodings()
            if len(new_persons) != len(known_persons):
                print(f"Updated: now watching {len(new_persons)} persons.")
            known_persons = new_persons

        # Decay all face states every frame
        stale_keys = []
        for key in face_states:
            face_states[key]['frames_left'] -= 1
            if face_states[key]['frames_left'] <= 0:
                stale_keys.append(key)
        for key in stale_keys:
            del face_states[key]

        # Detect faces every frame for smooth drawing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(80, 80)
        )

        # Run recognition only every 5th frame to reduce lag
        do_recognition = (frame_count % 5 == 0)

        for i, (x, y, w, h) in enumerate(faces):
            face_key = get_face_key(x, y, w, h)
            existing_key = find_matching_state(face_key, face_states)

            if do_recognition:
                # Crop face region with padding
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                face_crop = frame[y1:y2, x1:x2]

                temp_path = f"temp_face_{i}.jpg"
                cv2.imwrite(temp_path, face_crop)
                unknown_encoding = processor.extract_face_encoding(temp_path)

                if unknown_encoding is not None:
                    best_match = None
                    best_confidence = 0

                    for person in known_persons:
                        known_enc = person['encoding']
                        min_len = min(len(known_enc), len(unknown_encoding))
                        k = known_enc[:min_len]
                        u = unknown_encoding[:min_len]

                        dot = np.dot(k, u)
                        na = np.linalg.norm(k)
                        nb = np.linalg.norm(u)

                        if na == 0 or nb == 0:
                            continue

                        similarity = dot / (na * nb)
                        confidence = float(similarity * 100)

                        print(f"Face {i} @ {face_key} - Checking {person['name']}: {confidence:.1f}%")

                        if similarity >= 0.70 and confidence > best_confidence:
                            best_match = person
                            best_confidence = confidence

                    # Store result by position key (not index)
                    state_key = existing_key if existing_key else face_key
                    if best_match is not None:
                        face_states[state_key] = {
                            'match': best_match,
                            'confidence': best_confidence,
                            'frames_left': stay_green_frames
                        }
                    else:
                        face_states[state_key] = {
                            'match': None,
                            'confidence': 0,
                            'frames_left': stay_green_frames // 2
                        }

                if os.path.exists(temp_path):
                    os.remove(temp_path)

            # Draw result using position-based state
            draw_key = existing_key if existing_key else face_key
            state = face_states.get(draw_key, {})

            if state.get('match') and state.get('frames_left', 0) > 0:
                # GREEN — matched person
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                label = f"FOUND: {state['match']['name']} ({state['confidence']:.1f}%)"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Save alert with cooldown
                if do_recognition:
                    person_id = state['match']['id']
                    now = datetime.now()
                    should_save = True

                    if person_id in alert_cooldown:
                        time_diff = (now - alert_cooldown[person_id]).seconds
                        if time_diff < cooldown_seconds:
                            should_save = False

                    if should_save:
                        os.makedirs('alerts', exist_ok=True)
                        snapshot_name = f"alert_{person_id}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                        snapshot_path = os.path.join('alerts', snapshot_name)
                        cv2.imwrite(snapshot_path, frame)
                        save_alert(person_id, state['confidence'], snapshot_path)
                        alert_cooldown[person_id] = now
                        print(f"Alert saved! Snapshot: {snapshot_path}")
            else:
                # RED — unknown
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Status bar
        status = f"Watching {len(known_persons)} persons | Faces: {len(faces)}"
        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Missing Person Detection - Press Q to quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


if __name__ == '__main__':
    run_camera()