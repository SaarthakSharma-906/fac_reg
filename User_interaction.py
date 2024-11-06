import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np

# Import your feature extraction and inference functions
from feature_extraction import FaceFeaturesExtractor
from joblib import load

# Load the model
model, classes = load('face_recogniser_on_aug_data_1.pkl')

# Initialize face extractor
features_extractor = FaceFeaturesExtractor()

def capture_image():
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture image")
        return

    # Convert frame to PIL image for preprocessing
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Extract face embeddings
    bbs, embeddings = features_extractor(img)
    if embeddings is None:
        messagebox.showinfo("Info", "No face detected!")
        return

    # Set a threshold to consider embeddings as similar (tune as needed)
    similarity_threshold = 0.6
    processed_indices = set()
    results = []

    for i, embedding in enumerate(embeddings):
        if embedding.shape[0] != 512:
            print(f"Skipping face with invalid embedding size: {embedding.shape}")
            continue

        # Skip this face if it’s too similar to any previously processed face
        is_similar = any(
            np.dot(embedding, embeddings[j]) > similarity_threshold
            for j in processed_indices
        )
        if is_similar:
            continue

        # Process this face and add its index to processed
        person_index = model.predict([embedding.flatten()])[0]
        person_name = classes[person_index]
        results.append(person_name)
        processed_indices.add(i)

    result_text = "Persons Identified: " + ", ".join(results) if results else "No valid faces detected for prediction"
    messagebox.showinfo("Result", result_text)



# Function to display the camera feed
def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        label.config(image=img)
        label.image = img
    label.after(10, update_frame)

# Initialize GUI
root = tk.Tk()
root.title("Real-Time Face Recognition")
label = tk.Label(root)
label.pack()

# Button to capture image
capture_button = tk.Button(root, text="Capture", command=capture_image)
capture_button.pack()

# Start the camera
cap = cv2.VideoCapture(0)
update_frame()

# Start the GUI loop
root.mainloop()

# Release the camera when closing
cap.release()
cv2.destroyAllWindows()
