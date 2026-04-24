import pickle
import cv2
import numpy as np
import mysql.connector
from face_processor import FaceProcessor

processor = FaceProcessor()

# Load from database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="admin",
    database="missing_persons_db"
)
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT full_name, face_encoding FROM missing_persons")
persons = cursor.fetchall()
print(f"Found {len(persons)} persons in database")

# Capture webcam frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

# Detect face in frame exactly like camera_recognition.py does
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

print(f"Faces detected in webcam: {len(faces)}")

if len(faces) == 0:
    print("No face detected in webcam frame!")
    print("Trying with full frame instead...")
    cv2.imwrite("test_crop.jpg", frame)
else:
    x, y, w, h = faces[0]
    face_crop = frame[y:y+h, x:x+w]
    cv2.imwrite("test_crop.jpg", face_crop)
    print(f"Face cropped: {w}x{h} pixels")

# Get encoding of cropped face
webcam_encoding = processor.extract_face_encoding("test_crop.jpg")
print(f"Webcam encoding length: {len(webcam_encoding)}")

# Compare
print()
for person in persons:
    db_encoding = pickle.loads(person["face_encoding"])

    min_len = min(len(db_encoding), len(webcam_encoding))
    db_enc = db_encoding[:min_len]
    wc_enc = webcam_encoding[:min_len]

    dot = np.dot(db_enc, wc_enc)
    na = np.linalg.norm(db_enc)
    nb = np.linalg.norm(wc_enc)
    similarity = dot / (na * nb)

    print(f"Person: {person['full_name']}")
    print(f"DB encoding length: {len(db_encoding)}")
    print(f"Similarity: {similarity * 100:.1f}%")
    print(f"Will match at 0.80 threshold: {similarity >= 0.80}")
    print()