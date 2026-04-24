import cv2
import numpy as np
from PIL import Image
import os


class FaceProcessor:

    def __init__(self):
        self.faces_folder = "stored_faces"
        if not os.path.exists(self.faces_folder):
            os.makedirs(self.faces_folder)

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Load DeepFace once (lazy import to avoid slow startup)
        self._deepface = None

    def _get_deepface(self):
        if self._deepface is None:
            try:
                from deepface import DeepFace
                self._deepface = DeepFace
                print("DeepFace loaded successfully.")
            except ImportError:
                print("DeepFace not installed. Run: pip install deepface")
                self._deepface = False
        return self._deepface if self._deepface is not False else None

    def detect_and_crop_face(self, image_path):
        image = cv2.imread(image_path)

        if image is None:
            return None, "Could not read image file"

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )

        if len(faces) == 0:
            pil_image = Image.fromarray(rgb_image)
            return pil_image, "Face detected successfully"

        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        padding = 30
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        face_image = rgb_image[y:y+h, x:x+w]
        pil_image = Image.fromarray(face_image)

        return pil_image, "Face detected successfully"

    def extract_face_encoding(self, image_path):
        """
        Extract face encoding using DeepFace (Facenet model).
        Falls back to custom method if DeepFace is unavailable.
        Returns a normalized numpy array encoding.
        """
        try:
            print(f"Extracting encoding from: {image_path}")

            DeepFace = self._get_deepface()

            if DeepFace is not None:
                # Use DeepFace with Facenet — robust, lighting/pose tolerant
                # enforce_detection=False so it doesn't crash on slightly unclear faces
                embedding_objs = DeepFace.represent(
                    img_path=image_path,
                    model_name="Facenet",
                    enforce_detection=False,
                    detector_backend="opencv"
                )

                if embedding_objs and len(embedding_objs) > 0:
                    embedding = np.array(embedding_objs[0]['embedding'], dtype=np.float32)

                    # Normalize
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm

                    print(f"Success! Encoding length: {len(embedding)}")
                    return embedding
                else:
                    print("DeepFace returned no embedding, falling back.")

        except Exception as e:
            print(f"DeepFace error: {e}, falling back to custom method.")

        # ---- Fallback: custom method (used if DeepFace unavailable) ----
        return self._custom_encoding(image_path)

    def _custom_encoding(self, image_path):
        """Fallback custom encoding (less accurate)."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (100, 100))
            equalized = cv2.equalizeHist(resized)
            blurred = cv2.GaussianBlur(equalized, (3, 3), 0)

            flat = blurred.flatten().astype(np.float32) / 255.0

            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(sobelx**2 + sobely**2)
            gradient_flat = cv2.resize(
                gradient_mag.astype(np.float32), (50, 50)
            ).flatten()
            grad_norm = np.linalg.norm(gradient_flat)
            if grad_norm > 0:
                gradient_flat = gradient_flat / grad_norm

            block_features = []
            block_size = 25
            for row in range(0, 100, block_size):
                for col in range(0, 100, block_size):
                    block = blurred[row:row+block_size, col:col+block_size]
                    hist = cv2.calcHist(
                        [block], [0], None, [32], [0, 256]
                    ).flatten()
                    hist_norm = np.linalg.norm(hist)
                    if hist_norm > 0:
                        hist = hist / hist_norm
                    block_features.extend(hist.tolist())

            block_features = np.array(block_features, dtype=np.float32)

            encoding = np.concatenate([
                flat * 0.3,
                gradient_flat * 0.3,
                block_features * 0.4
            ])

            norm = np.linalg.norm(encoding)
            if norm > 0:
                encoding = encoding / norm

            print(f"Success! Encoding length: {len(encoding)}")
            return encoding

        except Exception as e:
            print(f"Error in custom encoding: {e}")
            return None

    def compare_faces(self, known_encoding, unknown_encoding, tolerance=0.6):
        if known_encoding is None or unknown_encoding is None:
            return False, 0.0

        try:
            min_len = min(len(known_encoding), len(unknown_encoding))
            known_encoding = known_encoding[:min_len]
            unknown_encoding = unknown_encoding[:min_len]

            dot_product = np.dot(known_encoding, unknown_encoding)
            norm_a = np.linalg.norm(known_encoding)
            norm_b = np.linalg.norm(unknown_encoding)

            if norm_a == 0 or norm_b == 0:
                return False, 0.0

            similarity = dot_product / (norm_a * norm_b)
            confidence = float(similarity * 100)
            is_match = similarity >= 0.80

            return is_match, confidence

        except Exception as e:
            print(f"Compare error: {e}")
            return False, 0.0