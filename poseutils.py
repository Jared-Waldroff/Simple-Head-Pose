import mediapipe as mp
import numpy as np


class LMinfo:
    """
    A class that stores important facial landmarks and their corresponding indices.
    """

    def __init__(self):
        self.NOSE = 1
        self.FOREHEAD = 10
        self.LEFT_EYE = 33
        self.MOUTH_LEFT = 61
        self.CHIN = 199
        self.RIGHT_EYE = 263
        self.MOUTH_RIGHT = 291

    def get(self):
        """
        Returns the list of key landmark indices.
        """
        return [self.NOSE,
                self.FOREHEAD,
                self.LEFT_EYE,
                self.MOUTH_LEFT,
                self.CHIN,
                self.RIGHT_EYE,
                self.MOUTH_RIGHT]


class Scaler:
    """
    A class responsible for scaling and normalizing facial landmarks.
    """

    def scale(self, landmarks):
        """
        Scales the landmarks based on the distance between the forehead and chin.
        Center landmarks around the nose point.

        Args:
        - landmarks: List of (x, y) tuples representing facial landmarks.

        Returns:
        - Normalized and scaled landmarks.
        """
        nose_point = landmarks[0]
        landmarks = [lm - nose_point for lm in landmarks]

        # Calculate distance between forehead and chin for scaling
        forehead_point = landmarks[1]
        chin_point = landmarks[4]
        reference_length = np.linalg.norm(forehead_point - chin_point)

        if reference_length == 0:
            raise ValueError("Invalid reference length. The landmarks may not be properly detected.")

        # Scale landmarks by reference length
        landmarks = landmarks / reference_length
        return landmarks


class MPMesher:
    """
    A class that wraps the MediaPipe FaceMesh model for landmark extraction.
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.landmarks_info = LMinfo()
        self.mesher = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, image):
        """
        Processes an image and extracts key facial landmarks.

        Args:
        - image: The input image.

        Returns:
        - A flattened list of (x, y) landmark coordinates, or None if no landmarks are detected.
        """
        results = self.mesher.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in self.landmarks_info.get():
                        landmarks.append((lm.x, lm.y))
                landmarks_flattened = [item for tup in landmarks for item in tup]
                return landmarks_flattened
        return None


class Mesher:
    """
    A class that serves as a general wrapper for different mesher types.
    Default mesher is MediaPipe FaceMesh.
    """

    def __init__(self, mesher_type="mp", min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mesher = self.__initmesher(mesher_type, min_detection_confidence, min_tracking_confidence)

    def __initmesher(self, mesher_type, min_detection_confidence, min_tracking_confidence):
        """
        Initializes the appropriate mesher based on the provided mesher_type.
        """
        if mesher_type == "mp":
            return MPMesher(min_detection_confidence, min_tracking_confidence)
        else:
            raise ValueError(f"Unsupported mesher type: {mesher_type}")

    def process(self, image):
        """
        Extracts landmarks from the given image using the initialized mesher.

        Args:
        - image: The input image.

        Returns:
        - A flattened list of (x, y) landmark coordinates, or None if no landmarks are detected.
        """
        return self.mesher.process(image)
