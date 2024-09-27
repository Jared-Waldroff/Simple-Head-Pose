import hpe
import cv2
import os

# Initialize the head pose estimation model
model = hpe.SimplePose(model_type="svr")

# Check if the model exists before loading
model_path = "best_model_svr_23_01_24_17"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure the model is available.")
else:
    # Load the pretrained model
    model.load(model_path)

# Define the image path
image_path = "examples/faces_3.png"

# Check if the image exists before loading
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file '{image_path}' not found. Please ensure the image is available.")
else:
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Predict poses, landmarks, and bounding boxes
    poses, lms, bbox = model.predict(image)

    # Draw the results on the image
    drawn_image = model.draw(image, poses, lms, bbox, draw_face=True, draw_person=False, draw_axis=True)

    # Display the result
    cv2.imshow("Image", drawn_image)

    # Save the result as an example image
    cv2.imwrite("example.png", drawn_image)

    # Wait for key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
