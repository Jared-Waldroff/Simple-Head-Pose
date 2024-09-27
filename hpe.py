import modelhub
import dataset
import drawer
import detector
import poseutils
import cv2
import os


# Class that encapsulates the HPE model and its functionalities,
# executing the entire pipeline to train, test and predict the pose parameters
class SimplePose:

    def __init__(self, model_type="svr", mesh_type="mp", mesh_conf=0.25, mesh_iou=0.45, yolo_conf=0.25, yolo_iou=0.45):
        """
        Initialize the HPE model with default parameters.
        """
        # Load the model from modelhub
        self.model = modelhub.load(
            model_name=model_type,
            mesh_type=mesh_type,
            mesh_conf=mesh_conf,
            mesh_iou=mesh_iou
        )

        # Initialize dataset manager, detector, and drawer objects
        self.dataset = dataset.DataManager()
        self.detector = detector.Detector(yolo_conf, yolo_iou)
        self.drawer = drawer.Drawer()

    def train(self, dataset_folder, save=True, split=0.1, ext="jpg"):
        """
        Train the model with a given dataset folder.
        """
        # Assign dataset path and load data
        self.dataset.assign_path(dataset_folder)
        x, y = self.dataset.load(ext=ext)

        # Split the dataset and train the model
        X_train, X_test, y_train, y_test = self.dataset.train_test_split(x, y, test_size=split)
        self.model.train(X_train, y_train, save=save)
        self.model.eval(X_test, y_test)

    def load(self, model_name):
        """
        Load a pretrained model.
        """
        self.model.load(model_name)

    def predict(self, image):
        """
        Predict pose parameters for a given image.
        """
        # Load and process image if a path is provided
        if isinstance(image, str):
            if os.path.exists(image):
                image = cv2.imread(image)
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            else:
                raise FileNotFoundError(f"Image file {image} not found.")

        # Detect faces and persons in the image
        landmarks, poses, bbox = [], [], {}
        faces, bbox = self.detector.detect(image)

        for face in faces:
            # Extract face id and image
            key = list(face.keys())[0]
            face = face[key]

            # Predict pose and landmarks for the detected face
            pose, lms = self.model.predict(face, return_landmarks=True)
            scaled_lms = None

            # Scale landmarks back to the original image size
            if lms is not None:
                lms = [(lms[i], lms[i + 1]) for i in range(0, len(lms) - 1, 2)]
                scaled_lms = [(int(lm[0] * face.shape[1]) + bbox[key]["face"][0],
                               int(lm[1] * face.shape[0]) + bbox[key]["face"][1]) for lm in lms]

            landmarks.append(scaled_lms)
            poses.append(pose)

        return poses, landmarks, bbox

    def draw(self, image, poses, lms, bbox, axis_size=50, draw_face=True, draw_person=False, draw_lm=False,
             draw_axis=True):
        """
        Draw results on the image using the drawer module.
        """
        if draw_person or draw_face:
            image = self.drawer.draw_bbox(image, bbox, draw_face=draw_face, draw_person=draw_person)

        if draw_lm:
            image = self.drawer.draw_landmarks(image, lms)

        if draw_axis:
            image = self.drawer.draw_axis(cv2.flip(image, 1), poses, lms, bbox, axis_size)

        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# Test Block (Optional)
if __name__ == "__main__":
    pose_model = SimplePose()
    # Replace 'dataset_folder' with the path to your dataset and 'model_name' with the model file name
    dataset_folder = "path_to_your_dataset"
    model_name = "your_model_name"

    # Train the model if dataset is available
    if os.path.exists(dataset_folder):
        pose_model.train(dataset_folder, save=True)

    # Load a pre-trained model for testing
    pose_model.load(model_name)

    # Replace 'image_path' with the path to your test image
    image_path = "path_to_your_image.jpg"
    if os.path.exists(image_path):
        poses, landmarks, bbox = pose_model.predict(image_path)
        drawn_image = pose_model.draw(image_path, poses, landmarks, bbox)
        cv2.imshow("Pose Estimation", drawn_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Test image not found at: {image_path}")
