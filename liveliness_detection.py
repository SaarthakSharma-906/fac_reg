import cv2
from deepface import DeepFace

def detect_liveness(frame):
    # Check liveness using DeepFace
    try:
        # Run the liveness check directly on the captured frame
        result = DeepFace.analyze(img_path=frame, actions=['emotion', 'age', 'gender', 'race'], detector_backend='mtcnn')
        
        # Print the results
        print(f"Analysis Result: {result}")
        print("Emotion:", result['dominant_emotion'])
        print("Age:", result['age'])
        print("Gender:", result['gender'])
        print("Race:", result['dominant_race'])

    except Exception as e:
        print(f"Error occurred during liveness detection: {e}")

def detect_liveness_from_camera():
    cap = cv2.VideoCapture(0)  # Change to the index of your Pi Camera if needed

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera")
            break

        # Display the live feed
        cv2.imshow('Live Feed', frame)

        # Perform liveness detection on the frame
        detect_liveness(frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the liveness detection from the camera
detect_liveness_from_camera()
