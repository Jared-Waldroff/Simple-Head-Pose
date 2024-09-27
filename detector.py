import torch
import cv2
import numpy as np


class Detector:
    def __init__(self, confidence=0.25, iou=0.45):
        try:
            # Load the YOLOv5 generic model for person detection
            self.pmodel = torch.hub.load('ultralytics/yolov5', 'yolov5l')

            # Load the YOLOv5 model for face detection trained on WIDER dataset
            self.fmodel = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/yolov5_faces.pt')

            # Setting the confidence and IOU thresholds
            self.pmodel.conf = confidence
            self.pmodel.iou = iou
            self.fmodel.conf = confidence
            self.fmodel.iou = iou
        except Exception as e:
            print(f"Error loading models: {e}")

    def set_person_threshold(self, confidence, iou):
        self.pmodel.conf = confidence
        self.pmodel.iou = iou

    def set_face_threshold(self, confidence, iou):
        self.fmodel.conf = confidence
        self.fmodel.iou = iou

    def detect(self, image_rgb):
        results = self.pmodel(image_rgb)
        class_labels = results.pandas().xyxy[0]['name'].tolist()
        bbox_coordinates = results.pandas().xyxy[0][['xmin', 'ymin', 'xmax', 'ymax']].values

        faces_imgs = []
        bb_dict = {}
        dict_id = 0

        for i in range(len(class_labels)):
            if class_labels[i] == 'person':
                bb_dict[dict_id] = {}
                px1, py1, px2, py2 = bbox_coordinates[i]
                bb_dict[dict_id]["person"] = [px1, py1, px2, py2]
                subimage = image_rgb[int(py1):int(py2), int(px1):int(px2)]

                try:
                    fresults = self.fmodel(subimage)
                except Exception as e:
                    print(f"Error processing face model: {e}")
                    continue

                face_bbox_coordinates = fresults.xyxy[0][:, :4].detach().numpy()
                for bbox in face_bbox_coordinates:
                    fx1, fy1, fx2, fy2 = bbox
                    bb_dict[dict_id]["face"] = [fx1 + px1, fy1 + py1, fx2 + px1, fy2 + py1]
                    faceimage = subimage[int(fy1):int(fy2), int(fx1):int(fx2)]
                    faces_imgs.append({dict_id: faceimage})

                dict_id += 1

        return faces_imgs, bb_dict


# Class tester code block
if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    facedet = Detector()
    faces, bb_dict = facedet.detect(image_rgb)

    print(bb_dict)
    for face in faces:
        for face_id, face_img in face.items():
            cv2.imshow(f'Face {face_id}', face_img)
            cv2.waitKey(0)

    for key in bb_dict.keys():
        print(bb_dict[key])
        px1, py1, px2, py2 = bb_dict[key]["person"]
        cv2.rectangle(image, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), 2)

        if "face" in bb_dict[key]:
            fx1, fy1, fx2, fy2 = bb_dict[key]["face"]
            cv2.rectangle(image, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (0, 255, 0), 2)

    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
