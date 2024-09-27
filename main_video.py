import hpe
import cv2
import os
import torch

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the head pose estimation model
model = hpe.SimplePose(model_type="svr")

# Check if the model exists before loading
model_path = os.path.join("trained", "best_model_svr_23_01_24_17")
if not os.path.exists(model_path + ".joblib"):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure the model is available.")
else:
    # Load the pretrained model
    model.load("best_model_svr_23_01_24_17")

# Move the model to the GPU if available
model.to(device)

# Function to process a video file and save the annotated output
def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open input video '{input_video_path}'.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video or failed to capture frame.")
            break

        # Flip the image horizontally and convert the color space from BGR to RGB
        image_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # Move the image to the GPU if available
        image_rgb_tensor = torch.from_numpy(image_rgb).to(device)

        # Get pose predictions and bounding boxes
        poses, lms, bbox = model.predict(image_rgb_tensor)

        # Draw the results on the image
        drawn_image = model.draw(image_rgb, poses, lms, bbox, draw_face=True, draw_person=False, draw_axis=True)

        display_image = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)
        out.write(display_image)
        cv2.imshow("Pose Estimation", display_image)

        if cv2.waitKey(5) & 0xFF == 27:
            print("Exiting...")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Output saved at: {output_video_path}")

if __name__ == "__main__":
    input_video_path = "input_video.mp4"
    output_video_path = "output_video.mp4"
    process_video(input_video_path, output_video_path)
