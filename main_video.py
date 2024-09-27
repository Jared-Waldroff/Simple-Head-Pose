import hpe
import cv2
import os

# Initialize the head pose estimation model
model = hpe.SimplePose(model_type="svr")

# Check if the model exists before loading
model_path = os.path.join("trained", "best_model_svr_23_01_24_17")
if not os.path.exists(model_path + ".joblib"):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure the model is available.")
else:
    # Load the pretrained model
    model.load("best_model_svr_23_01_24_17")


def process_video(input_video_path, output_video_path):
    """
    Processes the input video frame-by-frame using the head pose estimation model
    and saves the output video with annotations.
    """
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open input video '{input_video_path}'.")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set up the video writer
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video or failed to capture frame.")
            break

        # Convert the color space from BGR to RGB (without flipping)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get pose predictions and bounding boxes
        poses, lms, bbox = model.predict(image_rgb)

        # Draw the results on the image
        drawn_image = model.draw(image_rgb, poses, lms, bbox, draw_face=True, draw_person=False, draw_axis=True)

        # Convert the image back to BGR for saving and display in OpenCV
        display_image = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)

        # Write the frame to the output video
        out.write(display_image)

        # Display the resulting frame (optional)
        cv2.imshow("Pose Estimation", display_image)

        # Break the loop if the user presses the ESC key
        if cv2.waitKey(5) & 0xFF == 27:
            print("Exiting...")
            break

    # Release the video source and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Output saved at: {output_video_path}")


if __name__ == "__main__":
    # Input video file path (change this to your input video file path)
    input_video_path = "C:/Users/jared/PycharmProjects/Simple-Head-Pose/assets/classLow.mp4"  # Replace with your input video path
    output_video_path = "output_video.mp4"  # Replace with your desired output video path

    # Process the input video and save the output
    process_video(input_video_path, output_video_path)
