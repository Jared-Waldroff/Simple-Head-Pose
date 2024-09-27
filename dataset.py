from sklearn.model_selection import train_test_split
import pandas as pd
import mediapipe as mp
import poseutils  # Ensure this module is available in your environment
import scipy.io
import tqdm
import json
import cv2
import os


class DataManager:
    def __init__(self, datapath=""):
        self.datapath = datapath
        self.landmarks_info = poseutils.LMinfo()
        self.scaler = poseutils.Scaler()

        # Initialize MediaPipe Face Mesh
        self.mesher = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.25,
            min_tracking_confidence=0.45)

    def assign_path(self, datapath):
        self.datapath = datapath

    def get_landmarks(self, results):
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in self.landmarks_info.get():
                        landmarks.append((lm.x, lm.y))
                landmarks_flattened = [item for tup in landmarks for item in tup]
                return landmarks_flattened
        else:
            return None

    def jsonify(self, scale=True, ext="jpg"):
        skipped = 0
        data = {}

        for file in tqdm.tqdm(os.listdir(self.datapath)):
            keyname = file.split(".")[0]
            if keyname not in data:
                data[keyname] = {}
                data[keyname]["landmarks"] = []
                data[keyname]["angles"] = []

            if file.endswith(ext):
                image_path = os.path.join(self.datapath, file)  # Update with correct path
                image = cv2.imread(image_path)
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                results = self.get_landmarks(self.mesher.process(image))

                if results is None:
                    skipped += 1
                    continue

                if scale:
                    landmarks = self.scaler.scale(results)
                data[keyname]["landmarks"] = landmarks

                mat_path = os.path.join(self.datapath, keyname + ".mat")  # Update with correct path
                mat = scipy.io.loadmat(mat_path)
                pitch = float(mat["Pose_Para"][0][0])
                yaw = float(mat["Pose_Para"][0][1])
                roll = float(mat["Pose_Para"][0][2])
                data[keyname]["angles"] = [yaw, pitch, roll]

        print(
            f"Skipped {skipped} images due to no results on a total of {len(os.listdir(self.datapath)) / 2} images from folder")

        with open("dataset.json", "w") as outfile:
            json.dump(data, outfile)

    def load(self, scale=True, ext="jpg"):
        skipped = 0
        train_data = []
        target_data = []

        for file in tqdm.tqdm(os.listdir(self.datapath)):
            if file.endswith(ext):
                image_path = os.path.join(self.datapath, file)  # Update with correct path
                image = cv2.imread(image_path)
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                results = self.get_landmarks(self.mesher.process(image))

                if results is None:
                    skipped += 1
                    continue

                if scale:
                    landmarks = self.scaler.scale(results)

                mat_path = os.path.join(self.datapath, file.split('.')[0] + ".mat")  # Update with correct path
                mat = scipy.io.loadmat(mat_path)
                pitch = float(mat["Pose_Para"][0][0])
                yaw = float(mat["Pose_Para"][0][1])
                roll = float(mat["Pose_Para"][0][2])

                train_data.append(landmarks)
                target_data.append([yaw, pitch, roll])

        train_data = pd.DataFrame(train_data)
        target_data = pd.DataFrame(target_data)
        print(
            f"Skipped {skipped} images due to no results on a total of {len(os.listdir(self.datapath)) / 2} images from folder")

        return train_data, target_data

    def train_test_split(self, X, y, test_size=0.2, random_state=69):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
