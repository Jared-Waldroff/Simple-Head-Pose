from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint
from datetime import datetime
from sklearn.svm import SVR
from xgboost import XGBRegressor
import mediapipe as mp
import numpy as np
import poseutils
import joblib
import scipy
import cv2
import os


class Regressor:
    def __init__(self, regressor_type="svr", mesher_type="mp", mesh_conf=0.25, mesh_iou=0.45):
        """
        Initializes the Regressor with the specified regressor type and settings for the face mesh.
        """
        self.__init_model = self.__getinstance(regressor_type)
        self.model_path = "trained"
        self.model_type = regressor_type
        self.mesher = poseutils.Mesher(mesher_type, mesh_conf, mesh_iou)
        self.scaler = poseutils.Scaler()
        self.landmarks_info = poseutils.LMinfo()
        self.param_grid = self.__getparams(regressor_type)

    def __getinstance(self, regressor_type):
        """
        Returns the model instance based on the specified regressor type.
        """
        if regressor_type == "svr":
            return SVR()
        elif regressor_type == "xgboost":
            return XGBRegressor()
        else:
            raise ValueError(f"Unsupported regressor type: {regressor_type}")

    def __getparams(self, regressor_type):
        """
        Returns the hyperparameter grid for the specified regressor type.
        """
        if regressor_type == "svr":
            return {
                'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'estimator__C': np.arange(0.6, 0.75, 0.01),
                'estimator__gamma': np.arange(0.09, 0.1, 0.001),
                'estimator__epsilon': np.arange(0.07, 0.08, 0.001),
                'estimator__degree': [2, 3, 4]
            }
        elif regressor_type == "xgboost":
            return {
                "estimator__colsample_bytree": uniform(0.7, 0.3),
                "estimator__gamma": uniform(0, 0.5),
                "estimator__learning_rate": uniform(0.03, 0.3),
                "estimator__max_depth": randint(2, 6),
                "estimator__n_estimators": randint(100, 150),
                "estimator__subsample": uniform(0.6, 0.4)
            }
        else:
            raise ValueError(f"Unsupported regressor type: {regressor_type}")

    def set_rcparams(self, params):
        """
        Sets the parameter grid for randomized search.
        """
        self.param_grid = params

    def train(self, X_train, y_train, save=True):
        """
        Trains the model with the specified training data and hyperparameters.
        """
        mor = MultiOutputRegressor(self.__init_model)
        if self.model_type == "svr":
            self.search = GridSearchCV(mor, self.param_grid, scoring='neg_mean_squared_error', verbose=10, n_jobs=-1)
        elif self.model_type == "xgboost":
            self.search = RandomizedSearchCV(mor, self.param_grid, n_iter=20, verbose=10, n_jobs=-1)

        self.search.fit(X_train, y_train)
        self.model = self.search.best_estimator_

        if save:
            current_datetime = datetime.now().strftime("%d_%m_%y_%H")
            self.save(os.path.join(self.model_path, f"best_model_{self.model_type}_{current_datetime}"))

    def eval(self, X_val, y_val):
        """
        Evaluates the model on the validation set and prints the RMSE.
        """
        print("--------- EVALUATION RESULTS ---------")
        print(f"Best parameters found: {self.search.best_params_}")
        train_rmse = np.sqrt(-self.search.best_score_)
        print(f"train_rmse: {train_rmse}")
        validation_rmse = np.sqrt(mean_squared_error(y_val, self.model.predict(X_val)))
        print(f"validation_rmse: {validation_rmse}")
        print("--------------------------------------")

    def test(self, folder_path):
        """
        Tests the model on a set of images and prints the results.
        """
        for file in os.listdir(folder_path):
            if file.endswith(".jpg"):
                image = cv2.imread(os.path.join(folder_path, file))
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                results = self.mesher.process(image)
                if results is None:
                    print(f"Skipped {file} due to lack of results.")
                    continue
                landmarks = self.scaler.scale(results)
                predictions = self.model.predict([landmarks])
                mat = scipy.io.loadmat(os.path.join(folder_path, f"{file.split('.')[0]}.mat"))
                yaw, pitch, roll = float(mat["Pose_Para"][0][1]), float(mat["Pose_Para"][0][0]), float(
                    mat["Pose_Para"][0][2])
                targets = [yaw, pitch, roll]
                print(f"--------- Result for Image: {file} ---------")
                print(f"Predicted: {predictions[0]}")
                print(f"Target: {targets}")
                print("----------------------------------------")

    def predict(self, image, return_landmarks=False):
        """
        Predicts the pose parameters of a single image.
        """
        if isinstance(image, str):
            if os.path.exists(image):
                image = cv2.imread(image)
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            else:
                raise FileNotFoundError(f"Image file '{image}' not found.")

        results_lms = self.mesher.process(image)
        if results_lms is None:
            return (None, None) if return_landmarks else None

        scaled_lms = self.scaler.scale(results_lms)
        predictions = self.model.predict([scaled_lms])
        return (predictions[0], results_lms) if return_landmarks else predictions[0]

    def save(self, filename):
        """
        Saves the model to disk.
        """
        joblib.dump(self.model, filename + ".joblib")

    def load(self, filename):
        """
        Loads a model from disk.
        """
        full_path = os.path.join(self.model_path, filename + ".joblib")
        if os.path.exists(full_path):
            self.model = joblib.load(full_path)
        else:
            raise FileNotFoundError(f"Model file '{full_path}' not found.")
