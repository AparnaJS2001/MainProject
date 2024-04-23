import streamlit as st
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained violence detection model
violence_model_path = "MoBiLSTM_model.h5"  # Replace with the correct file path

# Check if the model file exists
if not os.path.exists(violence_model_path):
    st.error(f"Model file not found at {violence_model_path}. Please check the file path.")
else:
    violence_model = load_model(violence_model_path)

    # Function to preprocess and classify video frames
    def classify_video_frame(frame):
        # Preprocess the frame (resize, normalize, etc.)
        preprocessed_frame = preprocess_frame(frame)

        # Make a prediction using the trained model
        prediction = violence_model.predict(np.expand_dims(preprocessed_frame, axis=0))[0, 0]

        return prediction

    def preprocess_frame(frame):
        # Resize the frame to match the input size of your model
        resized_frame = cv2.resize(frame, (64, 64))
        # Normalize pixel values to be in the range [0, 1]
        normalized_frame = resized_frame / 255.0
        # Add a batch dimension and repeat the frame to have the shape (16, 64, 64, 3)
        preprocessed_frame = np.expand_dims(normalized_frame, axis=0)
        preprocessed_frame = np.repeat(preprocessed_frame, 16, axis=0)
        return preprocessed_frame

    # Streamlit app
    st.title("Violence Detection in Videos")

    # Option to upload a video file
    uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mkv"])

    # Option to record live video using webcam
    if st.button("Record Live Video"):
        st.write("Recording Live Video...")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Display frame and perform violence detection
            st.image(frame, channels="BGR", caption="Live Video")
            prediction = classify_video_frame(frame)
            if prediction > 0.5:
                st.success("No Violence Detected")
            else:
                st.error(" Violence Detected")

        # Release the video capture object
        cap.release()

    # Process the uploaded video file
    if uploaded_file is not None:
        st.video(uploaded_file)

        # Save the uploaded video file to a temporary location
        temp_file_path = "temp_video.mp4"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # OpenCV code for splitting video into frames and violence detection
        video_reader = cv2.VideoCapture(temp_file_path)

        # Calculate total number of frames in the video
        total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate step size for selecting frames
        step_size = max(total_frames // 10, 1)

        # Iterate through frames and display every 10th frame
        frame_count = 0
        while video_reader.isOpened():
            ret, frame = video_reader.read()
            if not ret:
                break

            # Display frame and perform violence detection
            if frame_count % step_size == 0:
                st.image(frame, channels="BGR", caption="Frame")
                prediction = classify_video_frame(frame)
                if prediction > 0.5:
                    st.success("No Violence Detected")
                else:
                    st.error("Violence Detected")

            frame_count += 1

        # Release the video capture object for the uploaded file
        video_reader.release()

        # Remove the temporary file
        os.remove(temp_file_path)
